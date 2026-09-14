import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import unittest


TRACER_DIR = Path(__file__).resolve().parents[1]
VERSION = "1.7.6"


class InstallNvbitTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project with spaces"
        self.project.mkdir()
        self.installer = self.project / "install_nvbit.sh"
        shutil.copy2(TRACER_DIR / "install_nvbit.sh", self.installer)
        self.sdk = self.project / "nvbit_release"
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.backups = self.root / "backups"
        self.backups.mkdir()
        self.events = self.root / "events.log"
        self.archive = self.root / "sdk.tar.bz2"
        self.env = os.environ.copy()
        self.env.update({
            "PATH": str(self.bin_dir) + os.pathsep + os.defpath,
            "TMPDIR": str(self.backups),
            "TEST_EVENTS": str(self.events),
            "TEST_ARCHIVE": str(self.archive),
            "TEST_ARCH": "x86_64",
        })
        self.write_command("uname", 'printf "%s\\n" "$TEST_ARCH"\n')
        self.write_command(
            "curl",
            'printf "download %s\\n" "$*" >> "$TEST_EVENTS"\n'
            'if [ "${TEST_DOWNLOAD_FAIL:-0}" = 1 ]; then exit 22; fi\n'
            'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = "-o" ]; then\n'
            '    cp "$TEST_ARCHIVE" "$2"\n'
            '    exit 0\n'
            '  fi\n'
            '  shift\n'
            'done\n'
            'exit 2\n',
        )
        self.write_archive()

    def write_command(self, name, body):
        path = self.bin_dir / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)

    def write_archive(self, version=VERSION, arch="x86_64", library=True):
        files = {
            "core/nvbit.h": '#define NVBIT_VERSION "{}"\n'.format(version),
        }
        if library:
            files["core/libnvbit.a"] = "test library"
        with tarfile.open(self.archive, "w:bz2") as archive:
            for name, contents in files.items():
                data = contents.encode()
                entry = tarfile.TarInfo("nvbit_release_" + arch + "/" + name)
                entry.size = len(data)
                entry.mtime = 1
                archive.addfile(entry, io.BytesIO(data))

    def install_existing(self, version):
        core = self.sdk / "core"
        core.mkdir(parents=True)
        (core / "nvbit.h").write_text(
            '#define NVBIT_VERSION "{}"\n'.format(version)
        )
        (core / "libnvbit.a").write_text("previous library")
        (self.sdk / "local-marker").write_text("previous SDK")

    def run_installer(self, **overrides):
        return subprocess.run(
            ["bash", str(self.installer)],
            cwd=self.root,
            env={**self.env, **overrides},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )

    def assert_no_staging(self):
        self.assertEqual(list(self.project.glob(".nvbit-install.*")), [])

    def test_fresh_install_and_noop_preserve_timestamps(self):
        before = time.time() - 1
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout)
        header = self.sdk / "core/nvbit.h"
        library = self.sdk / "core/libnvbit.a"
        self.assertIn(VERSION, header.read_text())
        self.assertGreaterEqual(header.stat().st_mtime, before)
        self.assertGreaterEqual(library.stat().st_mtime, before)
        self.assertIn(
            "/v1.7.6/nvbit-Linux-x86_64-1.7.6.tar.bz2",
            self.events.read_text(),
        )
        timestamps = (header.stat().st_mtime_ns, library.stat().st_mtime_ns)
        events = self.events.read_text()

        result = self.run_installer(TEST_DOWNLOAD_FAIL="1")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("already installed", result.stdout)
        self.assertEqual(self.events.read_text(), events)
        self.assertEqual(
            (header.stat().st_mtime_ns, library.stat().st_mtime_ns),
            timestamps,
        )
        self.assert_no_staging()

    def test_upgrade_preserves_old_sdk_and_invalidates_objects(self):
        self.install_existing("1.7.5")
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(VERSION, (self.sdk / "core/nvbit.h").read_text())
        backup = list(self.backups.glob("nvbit-backup.*/nvbit_release"))
        self.assertEqual(len(backup), 1)
        self.assertEqual(
            (backup[0] / "local-marker").read_text(), "previous SDK"
        )
        self.assertIn("1.7.5", (backup[0] / "core/nvbit.h").read_text())
        self.assertFalse((self.sdk / "local-marker").exists())
        self.assert_no_staging()

    def test_failed_download_preserves_existing_sdk(self):
        self.install_existing("1.7.5")
        result = self.run_installer(TEST_DOWNLOAD_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("1.7.5", (self.sdk / "core/nvbit.h").read_text())
        self.assertTrue((self.sdk / "local-marker").is_file())
        self.assertEqual(list(self.backups.iterdir()), [])
        self.assert_no_staging()

    def test_failed_placement_restores_existing_sdk(self):
        self.install_existing("1.7.5")
        self.write_command(
            "mv",
            'case "$1" in\n'
            '  */.nvbit-install.*/nvbit_release_*) exit 1 ;;\n'
            'esac\n'
            'exec /bin/mv "$@"\n',
        )
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("1.7.5", (self.sdk / "core/nvbit.h").read_text())
        self.assertTrue((self.sdk / "local-marker").is_file())
        self.assert_no_staging()

    def test_wrong_version_archive_preserves_existing_sdk(self):
        self.install_existing("1.7.5")
        self.write_archive(version="1.7.5")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wrong version", result.stdout)
        self.assertTrue((self.sdk / "local-marker").is_file())
        self.assert_no_staging()

    def test_incomplete_archive_is_not_installed(self):
        self.write_archive(library=False)
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.sdk.exists())
        self.assert_no_staging()

    def test_corrupt_archive_preserves_existing_sdk(self):
        self.install_existing("1.7.5")
        self.archive.write_bytes(b"not a tar archive")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.sdk / "local-marker").is_file())
        self.assert_no_staging()

    def test_aarch64_archive_selection(self):
        self.write_archive(arch="aarch64")
        result = self.run_installer(TEST_ARCH="aarch64")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "nvbit-Linux-aarch64-1.7.6.tar.bz2", self.events.read_text()
        )

    def test_unsupported_architecture_does_not_download(self):
        result = self.run_installer(TEST_ARCH="riscv64")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported host architecture", result.stdout)
        self.assertFalse(self.events.exists())
        self.assertFalse(self.sdk.exists())

    def prepare_make_tree(self):
        shutil.copy2(TRACER_DIR / "Makefile", self.project / "Makefile")
        self.installer.write_text(
            '#!/usr/bin/env bash\n'
            'printf "unexpected installer call\\n" >> "$TEST_EVENTS"\n'
            'exit 99\n'
        )
        for name in ("tracer_tool", "tracer_tool/traces-processing"):
            directory = self.project / name
            directory.mkdir(parents=True, exist_ok=True)
            directory.joinpath("Makefile").write_text(
                ".PHONY: all clean\n"
                "all:\n"
                '\t@test -s "$(TEST_SDK)/core/nvbit.h"\n'
                '\t@test -s "$(TEST_SDK)/core/libnvbit.a"\n'
                '\t@printf "build {}\\n" >> "$(TEST_EVENTS)"\n'
                "clean:\n"
                '\t@printf "clean {}\\n" >> "$(TEST_EVENTS)"\n'.format(
                    name, name
                )
            )
        self.env["TEST_SDK"] = str(self.sdk)

    def run_make(self, target=None):
        command = ["make", "-j4", "-C", str(self.project)]
        if target is not None:
            command.append(target)
        return subprocess.run(
            command,
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )

    def test_parallel_make_builds_without_invoking_installer(self):
        self.prepare_make_tree()
        self.install_existing(VERSION)
        result = self.run_make()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events.read_text().splitlines(), [
            "build tracer_tool",
            "build tracer_tool/traces-processing",
        ])

    def test_make_does_not_upgrade_existing_sdk(self):
        self.prepare_make_tree()
        self.install_existing("1.7.5")
        result = self.run_make()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("1.7.5", (self.sdk / "core/nvbit.h").read_text())
        self.assertTrue((self.sdk / "local-marker").is_file())
        self.assertEqual(list(self.backups.iterdir()), [])
        self.assertEqual(self.events.read_text().splitlines(), [
            "build tracer_tool",
            "build tracer_tool/traces-processing",
        ])

    def test_clean_does_not_install_sdk(self):
        self.prepare_make_tree()
        result = self.run_make("clean")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.sdk.exists())
        self.assertEqual(self.events.read_text().splitlines(), [
            "clean tracer_tool",
            "clean tracer_tool/traces-processing",
        ])

    def test_missing_sdk_does_not_trigger_installation(self):
        self.prepare_make_tree()
        result = self.run_make("all")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.sdk.exists())
        self.assertFalse(self.events.exists())

    def test_objects_depend_on_sdk_header(self):
        makefile = (TRACER_DIR / "tracer_tool/Makefile").read_text()
        self.assertIn("$(OBJECTS): $(NVBIT_PATH)/nvbit.h", makefile)


if __name__ == "__main__":
    unittest.main()
