from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain, CMakeDeps
from conan.tools.files import copy, load, rmdir
import os


class SynetConan(ConanFile):
    name = "synet"
    version_file = "prj/txt/UserVersion.txt"
    extra_files = ["3rd/Simd/prj/txt/UserVersion.txt"]

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
        "test": [
            "none", "inference_engine", "onnx", "precision",
            "performance_difference", "quantization", "stability",
            "optimizer", "bf16", "multi_threads", "video", "use_samples", "all",
        ],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "simd_optimizations": True,
        "bf16_round_test": False,
        "perf_level": 0,
        "python_wrapper": False,
        "test": "none",
    }

    _openvino_tests = ("inference_engine", "precision", "all")
    _onnxruntime_tests = ("onnx", "all")

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
        if str(self.options.test) in self._openvino_tests:
            self.requires("openvino/2026.0.2", visible=False)
        if str(self.options.test) in self._onnxruntime_tests:
            self.requires("onnxruntime/1.23.2", visible=False)
            self.requires("onnx/1.18.0", transitive_headers=True, visible=False)

    def package_id(self):
        _requires = [
            "simd", "cpl",
            # conditional:
            "openvino", "onnxruntime", "onnx",
        ]
        for name in _requires:
            if name in self.info.requires:
                self.info.requires[name].full_package_mode()

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
        tc.variables["SYNET_TEST"] = str(self.options.test)
        tc.variables["SYNET_USE_CONAN_PACKAGES"] = True
        tc.variables["SYNET_INFO"] = True
        tc.variables["SYNET_GET_VERSION"] = True
        tc.variables["CMAKE_CXX_STANDARD"] = "17"
        tc.variables["CMAKE_CXX_STANDARD_REQUIRED"] = "ON"
        compiler_executables = self.conf.get("tools.build:compiler_executables", default={})
        if compiler_executables.get("cpp"):
            tc.cache_variables["CMAKE_CXX_COMPILER"] = compiler_executables["cpp"]
        if compiler_executables.get("c"):
            tc.cache_variables["CMAKE_C_COMPILER"] = compiler_executables["c"]

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

        # Тестовые бинари
        if str(self.options.test) != "none":
            test_binaries = {
                "inference_engine": ["test_inference_engine"],
                "onnx": ["test_onnx"],
                "performance_difference": ["test_performance_difference"],
                "precision": ["test_precision"],
                "quantization": ["test_quantization"],
                "stability": ["test_stability"],
                "optimizer": ["test_optimizer"],
                "bf16": ["test_bf16"],
                "multi_threads": ["test_multi_threads"],
                "video": ["test_video"],
                "all": [
                    "test_inference_engine", "test_onnx", "test_performance_difference",
                    "test_precision", "test_quantization", "test_stability",
                    "test_optimizer", "test_bf16", "test_multi_threads", "test_video",
                ],
            }
            for binary in test_binaries.get(str(self.options.test), []):
                copy(self, binary, src=self.build_folder,
                     dst=os.path.join(self.package_folder, "bin"), keep_path=False)

        rmdir(self, os.path.join(self.package_folder, "share"))

    def package_info(self):
        self.cpp_info.libs = ["Synet"]
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.requires = ["simd::simd", "cpl::cpl"]

        if int(self.options.perf_level) >= 1:
            self.cpp_info.defines.append("SYNET_PERFORMANCE_STATISTIC")
        if self.options.bf16_round_test:
            self.cpp_info.defines.append("SYNET_BF16_ROUND_TEST")

        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.extend(["pthread", "m", "dl"])
