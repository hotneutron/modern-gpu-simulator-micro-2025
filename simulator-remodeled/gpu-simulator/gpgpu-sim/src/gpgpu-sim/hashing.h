// author: Mahmoud Khairy, (Purdue Univ)
// email: abdallm@purdue.edu

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "../option_parser.h"

#ifndef HASHING_H
#define HASHING_H

#include "../abstract_hardware_model.h"
#include "gpu-cache.h"

unsigned ipoly_hash_function(new_addr_type higher_bits, unsigned index,
                             unsigned bank_set_num);

unsigned bitwise_hash_function(new_addr_type higher_bits, unsigned index,
                               unsigned bank_set_num);

unsigned PAE_hash_function(new_addr_type higher_bits, unsigned index,
                           unsigned bank_set_num);

// Opt8/Opt9: balanced sub-partition hash for a NON-power-of-two slice count.
// IPoly only supports 2^n bank_set_num, so with 80 slices (40 channels x2, 40 is
// non-2^n) the sim hashes into 128 then folds with `% 80`, which double-counts
// slices 0..47 (pigeonhole 128->80) => up to 2:1 spatial load imbalance across L2
// slices (a pure sim artifact of the 40-channel config; HW slices are even to <=5%,
// verified via lts__cycles_active). This function bit-mixes the high address bits
// (avalanche) BEFORE the modulo, so `% n_slices` is uniform for ALL power-of-two
// strides (verified: cv<=0.015 for stride 128/1024/4096 vs the `%80` 2:1 bias),
// while staying deterministic + address-stable (same line -> same slice) and
// work-invariant (it only picks WHICH slice, never how many accesses). See
// L2_SLICE_PARALLELISM_H100.md section 8.4.
unsigned balanced_subpartition_hash(new_addr_type higher_bits, unsigned index,
                                    unsigned n_slices);

#endif
