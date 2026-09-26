import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "gpu-simulator/gpgpu-sim/src"
UNIT = SIM / "gpgpu-sim/remodeling/functional_unit.cc"


class TensorPipelineCapacityTest(unittest.TestCase):
    """Execute production tensor timing and FU issue/cycle code without CUDA."""

    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler not installed")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.binary = str(Path(cls.temp.name) / "tensor-pipeline")
        unit = UNIT.read_text()
        header = UNIT.with_suffix(".h").read_text()
        hardware = (SIM / "abstract_hardware_model.cc").read_text()
        hardware_header = (SIM / "abstract_hardware_model.h").read_text()

        def section(text, first, last):
            return text[text.index(first):text.index(last)]

        harness = r'''
#include <algorithm>
#include <cassert>
#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>
enum operation_pipeline_t {SP__OP, DP__OP, INTP__OP, SPECIALIZED__OP,
                          MEM__OP, SFU__OP, TENSOR_CORE__OP};
enum class TraceEnhancedOperandType { REG };
struct Register_file {};
struct Wait_Barrier_Entry_Modifier {};
struct shader_core_config {
  unsigned predicate_latency=13, warp_size=32;
  unsigned num_threads_granularity_read_regular_register_file_mem_inst=32;
  unsigned num_threads_granularity_read_regular_register_file_dp_inst=32;
  unsigned num_threads_granularity_read_regular_register_file_sfu_inst=32;
  unsigned num_threads_granularity_read_regular_register_file_other_inst=32;
  unsigned tensor_rate_per_cycle=32768;
  unsigned tensor_extra_latency_16816_fp32_1688_fp32=0;
  bool wgmma_step0_instrument_enable=false;
  const shader_core_config &get_gpgpu_sim_config() const {return *this;}
};
struct gpgpu_sim {
  shader_core_config config;
  const shader_core_config &get_config() const {return config;}
};
struct TensorInfo {
  bool is_set=true, is_sparse=false, is_16816_fp32_1688_fp32=false;
  unsigned size_m=64, size_n=128, size_k=16, operand_bit_size=16;
};
struct Extra {
  TensorInfo tensor;
  TensorInfo &get_tensor_core_instruction_info() {return tensor;}
};
struct warp_inst_t {
  explicit warp_inst_t(const shader_core_config * = nullptr) {}
  bool vacant=true, tensor=true;
  unsigned pc=0x100, id=0, mask=0xffffffff;
  unsigned latency=1, initiation_interval=1, latency_extra_predicate_op=0;
  unsigned cycles=0;
  operation_pipeline_t op_pipe=SPECIALIZED__OP;
  Extra extra;
  std::vector<uint64_t> addresses;
  bool empty() const {return vacant;}
  void clear() {vacant=true;}
  bool is_tensor_core_op() const {return tensor;}
  unsigned warp_id() const {return 7;}
  Extra &get_extra_trace_instruction_info() {return extra;}
  void generate_tensor_core_latencies(gpgpu_sim *);
'''
        harness += section(hardware_header, "  bool dispatch_delay() {",
                           "  void set_new_dispatch_delay(")
        harness += r'''
};
void move_warp_uniptr(std::unique_ptr<warp_inst_t> &dst,
                     std::unique_ptr<warp_inst_t> &src) {
  assert(dst && dst->empty() && src);
  dst.swap(src);
  src->clear();
}
struct register_set_uniptr {
  std::unique_ptr<warp_inst_t> inst{new warp_inst_t};
  warp_inst_t *get_ready() {return inst->empty()?nullptr:inst.get();}
  void move_out_to(std::unique_ptr<warp_inst_t> &dst) {move_warp_uniptr(dst,inst);}
};
struct Counter {void increment_with_integer(int) {}};
struct SM {
  shader_core_config config;
  struct {std::map<std::string,std::shared_ptr<Counter>> m_stats_map;} m_sm_stats;
  const shader_core_config *get_config() const {return &config;}
  void incsp_stat(unsigned,unsigned) {}
  void incmem_stat(unsigned,unsigned) {}
  void incsfu_stat(unsigned,unsigned) {}
  void inc_tensor_ops_for_warp(unsigned) {}
  void mark_consumer_warp(unsigned) {}
  void incexecstat(warp_inst_t *) {}
};
unsigned tick=0;
std::vector<std::tuple<unsigned,unsigned,unsigned>> retired;
'''
        harness += section(header, "class functional_unit {",
                           "struct functional_unit_with_queue_stage")
        harness += section(unit, "functional_unit::functional_unit(",
                           "void functional_unit::modify_wait_barrier_states(")
        harness += r'''
void functional_unit::release_read_barrier(std::unique_ptr<warp_inst_t> &) {}
void functional_unit::print(FILE *) const {}
bool functional_unit::instruction_finishing_execution(std::unique_ptr<warp_inst_t> &inst) {
  assert(inst->addresses==std::vector<uint64_t>({0x1000+inst->id,0x2000+inst->id}));
  assert(inst->mask==0xffffffff && m_active_insts_in_pipeline>0);
  retired.emplace_back(inst->id,tick,inst->latency);
  inst->clear();
  --m_active_insts_in_pipeline;
  return true;
}
'''
        harness += section(unit, "void functional_unit::cycle()",
                           "unsigned int functional_unit::get_active_lanes_in_pipeline()")
        harness += section(hardware, "void warp_inst_t::generate_tensor_core_latencies(",
                           "void warp_inst_t::assign_predicate_latencies_if_needed(")
        harness += r'''
struct Unit: functional_unit {
  Unit(unsigned depth, SM &sm, const std::string &name)
    :functional_unit(nullptr,nullptr,&sm.config,depth,name,&sm,SPECIALIZED__OP,
                     true,false,1,6,nullptr,1,false,TraceEnhancedOperandType::REG) {}
  unsigned depth() const {return m_pipeline_depth;}
  void submit(unsigned n, unsigned id, unsigned bits, bool sparse, unsigned extra,
              bool tensor=true, unsigned m=64, unsigned k=16) {
    register_set_uniptr input;
    input.inst->vacant=false; input.inst->id=id; input.inst->tensor=tensor;
    input.inst->extra.tensor.size_m=m;
    input.inst->extra.tensor.size_n=n;
    input.inst->extra.tensor.size_k=k;
    input.inst->extra.tensor.operand_bit_size=bits;
    input.inst->extra.tensor.is_sparse=sparse;
    input.inst->extra.tensor.is_16816_fp32_1688_fp32=extra!=0;
    gpgpu_sim gpu;
    gpu.config.tensor_extra_latency_16816_fp32_1688_fp32=extra;
    input.inst->generate_tensor_core_latencies(&gpu);
    input.inst->cycles=input.inst->initiation_interval;
    input.inst->addresses={0x1000+id,0x2000+id};
    reserve_unit(input);
    issue(input);
    assert(input.get_ready()==nullptr);
  }
  bool drained() const {return m_active_insts_in_pipeline==0 && m_dispatch_reg->empty();}
};
int main(int argc,char **argv) {
  assert(argc==9);
  unsigned initial=std::stoul(argv[1]), n=std::stoul(argv[2]), bits=std::stoul(argv[3]);
  bool sparse=std::stoul(argv[4]);
  unsigned extra=std::stoul(argv[5]);
  std::string mode=argv[6];
  unsigned m=std::stoul(argv[7]), k=std::stoul(argv[8]);
  SM sm;
  Unit unit(initial,sm,mode=="static"?"INT":"TENSOR");
  if(mode=="overlap") {
    unit.submit(128,1,16,false,0);
    for(tick=0;tick<32;++tick) unit.cycle();
    assert(unit.can_issue(nullptr));
    unit.submit(n,2,bits,sparse,extra,true,m,k);
  } else {
    unit.submit(n,1,bits,sparse,extra,mode!="static",m,k);
  }
  for(;tick<1024 && !unit.drained();++tick) unit.cycle();
  assert(unit.drained());
  std::cout<<"{\"depth\":"<<unit.depth()<<",\"retired\":[";
  bool first=true;
  for(const auto &event:retired) {
    if(!first) std::cout<<",";
    first=false;
    std::cout<<"["<<std::get<0>(event)<<","<<std::get<1>(event)<<","<<std::get<2>(event)<<"]";
  }
  std::cout<<"]}";
}
'''
        compiled = subprocess.run(
            [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
             "-Wno-unused-parameter",
             "-fsanitize=address,undefined", "-x", "c++", "-", "-o", cls.binary],
            input=harness, capture_output=True, text=True)
        if compiled.returncode:
            raise AssertionError(compiled.stderr)

    def execute(self, depth, n, bits=16, sparse=False, extra=0, mode="single",
                m=64, k=16):
        result = subprocess.run(
            [self.binary, str(depth), str(n), str(bits), str(int(sparse)), str(extra),
             mode, str(m), str(k)],
            capture_output=True, text=True)
        return result

    def test_subcycle_hmma_and_sparse_work_retires(self):
        # HMMA.1688.F16 is half a configured tensor cycle. Dense and sparse
        # positive work must survive the real dispatch and retirement path.
        for n, k, sparse, expected in [
                (8, 8, False, [[1, 1, 1]]),
                (8, 16, False, [[1, 1, 1]]),
                (8, 16, True, [[1, 1, 1]]),
                (40, 8, False, [[1, 2, 2]]),
                (40, 16, True, [[1, 2, 2]])]:
            with self.subTest(n=n, k=k, sparse=sparse):
                result = self.execute(32, n, sparse=sparse, m=16, k=k)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["retired"], expected)

    def test_shape_latency_fits_without_changing_retirement_time(self):
        for n, bits, sparse, extra in [
                (64, 16, False, 0), (128, 16, False, 0),
                (192, 16, False, 0), (256, 16, False, 0),
                (256, 16, True, 0), (192, 32, False, 0),
                (128, 16, False, 7)]:
            with self.subTest(n=n, bits=bits, sparse=sparse, extra=extra):
                compact = self.execute(32, n, bits, sparse, extra)
                reference = self.execute(256, n, bits, sparse, extra)
                self.assertEqual(compact.returncode, 0, compact.stderr)
                self.assertEqual(reference.returncode, 0, reference.stderr)
                observed = json.loads(compact.stdout)
                work_cycles = 64 * n * 16 * bits // 32768
                if sparse:
                    work_cycles //= 2
                latency = work_cycles - work_cycles // 2 + extra
                self.assertEqual(observed["depth"], max(32, latency))
                self.assertEqual(observed["retired"],
                                 [[1, work_cycles + 2 * extra - 1, latency]])
                self.assertEqual(observed["retired"],
                                 json.loads(reference.stdout)["retired"])

    def test_growing_preserves_older_inflight_instructions(self):
        compact = self.execute(32, 192, mode="overlap")
        reference = self.execute(256, 192, mode="overlap")
        self.assertEqual(compact.returncode, 0, compact.stderr)
        self.assertEqual(reference.returncode, 0, reference.stderr)
        self.assertEqual(json.loads(compact.stdout)["retired"],
                         json.loads(reference.stdout)["retired"])
        self.assertEqual(len(json.loads(compact.stdout)["retired"]), 2)

    def test_static_unit_configuration_mismatch_is_not_hidden(self):
        self.assertNotEqual(self.execute(32, 192, mode="static").returncode, 0)

    def test_reservation_capacity_is_still_bounded(self):
        result = self.execute(32, 2048)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exceeds reservation capacity", result.stderr)


if __name__ == "__main__":
    unittest.main()

# ## Revision History
# 2026-09-20 22:24 UTC: exercise subcycle HMMA, sparse rounding and fractional
# service work through production timing, dispatch and retirement code.
