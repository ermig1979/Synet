from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain, CMakeDeps
from conan.tools.files import copy, load, rmdir
import os


class SynetConan(ConanFile):
    name = "synet"

    # Package metadata
    description = "Synet Framework - neural network inference framework"
    topics = ("neural-networks", "inference", "simd", "performance")
    url = "https://github.com/ermig1979/Synet"
    license = "MIT"
    author = "Ermig1979"

    # Build settings and options
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "simd_optimizations": [True, False],  # SYNET_SIMD
        "bf16_round_test": [True, False],     # SYNET_BF16_ROUND_TEST
        "perf_level": [0, 1, 2],             # SYNET_PERF
        "python_wrapper": [True, False],     # SYNET_PYTHON
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "simd_optimizations": True,
        "bf16_round_test": False,
        "perf_level": 0,
        "python_wrapper": False,
    }

    def _repo_root(self):
        return os.path.normpath(os.path.join(self.recipe_folder, "..", ".."))

    def _read_version_file(self, filename, *subdirs):
        paths = [
            os.path.join(self._repo_root(), *subdirs, filename),
            os.path.join(self.recipe_folder, *subdirs, filename),
        ]
        for version_file in paths:
            if os.path.exists(version_file):
                version = load(self, version_file).strip()
                if version:
                    return version
        return None

    def set_version(self):
        if self.version:
            return
        version = self._read_version_file("UserVersion.txt", "prj", "txt")
        if version:
            self.version = version
            return
        raise Exception("Synet version not found in prj/txt/UserVersion.txt")

    def _read_simd_version(self):
        version = self._read_version_file("UserVersion.txt", "3rd", "Simd", "prj", "txt")
        if version:
            return version
        raise Exception(
            "Simd version not found. "
            "Run: git submodule update --init 3rd/Simd"
        )

    def export(self):
        copy(self, "UserVersion.txt",
             src=os.path.join(self._repo_root(), "prj", "txt"),
             dst=os.path.join(self.export_folder, "prj", "txt"))
        copy(self, "UserVersion.txt",
             src=os.path.join(self._repo_root(), "3rd", "Simd", "prj", "txt"),
             dst=os.path.join(self.export_folder, "3rd", "Simd", "prj", "txt"))

    def requirements(self):
        self.requires(f"simd/{self._read_simd_version()}")
        self.requires("cpl/1.0.0")

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")
        if int(self.options.perf_level) >= 2:
            self.options["simd"].perf = True

    def export_sources(self):
        project_root = os.path.join(self.recipe_folder, "..", "..")

        # Исходники src/
        copy(self, "*.cpp", src=os.path.join(project_root, "src"),
             dst=os.path.join(self.export_sources_folder, "src"))
        copy(self, "*.h", src=os.path.join(project_root, "src"),
             dst=os.path.join(self.export_sources_folder, "src"))

        # CMake файлы
        copy(self, "CMakeLists.txt", src=os.path.join(project_root, "prj", "cmake"),
             dst=os.path.join(self.export_sources_folder, "prj", "cmake"))
        copy(self, "*.cmake", src=os.path.join(project_root, "prj", "cmake"),
             dst=os.path.join(self.export_sources_folder, "prj", "cmake"))

        # Скрипты версионирования
        copy(self, "*.sh", src=os.path.join(project_root, "prj", "sh"),
             dst=os.path.join(self.export_sources_folder, "prj", "sh"))

        # Файлы версионирования
        copy(self, "*", src=os.path.join(project_root, "prj", "txt"),
             dst=os.path.join(self.export_sources_folder, "prj", "txt"))

        # Python wrapper
        copy(self, "*.py", src=os.path.join(project_root, "py"),
             dst=os.path.join(self.export_sources_folder, "py"))

        copy(self, "LICENSE", src=project_root, dst=self.export_sources_folder)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)

        tc.variables["SYNET_SHARED"] = self.options.shared
        tc.variables["SYNET_SIMD"] = self.options.simd_optimizations
        tc.variables["SYNET_PERF"] = int(self.options.perf_level)
        tc.variables["SYNET_PYTHON"] = self.options.python_wrapper
        tc.variables["SYNET_BF16_ROUND_TEST"] = self.options.bf16_round_test
        tc.variables["SYNET_TEST"] = "none"
        tc.variables["SYNET_INFO"] = True
        tc.variables["SYNET_GET_VERSION"] = True
        tc.variables["CMAKE_CXX_STANDARD"] = "17"
        tc.variables["CMAKE_CXX_STANDARD_REQUIRED"] = "ON"

        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)

        if os.path.exists(os.path.join(self.source_folder, "prj", "cmake", "CMakeLists.txt")):
            cmake_dir = os.path.join(self.source_folder, "prj", "cmake")
        else:
            cmake_dir = os.path.join(self.source_folder, "..", "..", "prj", "cmake")

        cmake.configure(build_script_folder=cmake_dir)
        cmake.build()

    def package(self):
        copy(self, "LICENSE", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))

        # Библиотека
        copy(self, "*.a", src=self.build_folder,
             dst=os.path.join(self.package_folder, "lib"), keep_path=False)
        copy(self, "*.so*", src=self.build_folder,
             dst=os.path.join(self.package_folder, "lib"), keep_path=False)
        copy(self, "*.lib", src=self.build_folder,
             dst=os.path.join(self.package_folder, "lib"), keep_path=False)
        copy(self, "*.dll", src=self.build_folder,
             dst=os.path.join(self.package_folder, "bin"), keep_path=False)

        # Заголовки Synet
        copy(self, "*.h",
             src=os.path.join(self.source_folder, "src", "Synet"),
             dst=os.path.join(self.package_folder, "include", "Synet"),
             keep_path=True)

        # Python файлы
        if self.options.python_wrapper:
            copy(self, "*.py",
                 src=os.path.join(self.source_folder, "py", "SynetPy"),
                 dst=os.path.join(self.package_folder, "python"),
                 keep_path=False)

        rmdir(self, os.path.join(self.package_folder, "share"))

    def package_info(self):
        self.cpp_info.libs = ["Synet"]
        self.cpp_info.includedirs = ["include"]

        if int(self.options.perf_level) >= 1:
            self.cpp_info.defines.append("SYNET_PERFORMANCE_STATISTIC")
        if self.options.bf16_round_test:
            self.cpp_info.defines.append("SYNET_BF16_ROUND_TEST")

        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.extend(["pthread", "m", "dl"])
