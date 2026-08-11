using System.Buffers.Binary;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Windows.AI.MachineLearning;

internal static class Program
{
    private static readonly int[] ExpectedInputShape = [1, 3, 640, 640];
    private static readonly int[] ExpectedOutputShape = [1, 84, 8400];
    private const int ExpectedInputElements = 1 * 3 * 640 * 640;
    private const int ExpectedOutputElements = 1 * 84 * 8400;
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public static async Task<int> Main(string[] args)
    {
        string? reportPath = FindOption(args, "--report");
        try
        {
            Options options = Options.Parse(args);
            reportPath = options.Report;
            await RunAsync(options);
            return 0;
        }
        catch (Exception error)
        {
            if (!string.IsNullOrWhiteSpace(reportPath))
            {
                WriteJson(reportPath, new
                {
                    schemaVersion = 1,
                    state = "failed",
                    runtimeExecuted = false,
                    runtimeIntrospectionComplete = false,
                    windowsMlApiCalled = false,
                    error = new { type = error.GetType().FullName, message = error.Message },
                    host = HostSnapshot(),
                });
            }
            Console.Error.WriteLine(error);
            return 1;
        }
    }

    private static async Task RunAsync(Options options)
    {
        if (!OperatingSystem.IsWindows())
        {
            throw new PlatformNotSupportedException("Windows ML spike runner must execute on Windows, not Linux ORT.");
        }

        Architecture architecture = RuntimeInformation.ProcessArchitecture;
        string target = architecture switch
        {
            Architecture.X64 => "win-x64",
            Architecture.Arm64 => "win-arm64",
            _ => throw new PlatformNotSupportedException($"Unsupported runner architecture: {architecture}"),
        };

        string modelSha256 = Sha256File(options.Model);
        if (!modelSha256.Equals(options.ExpectedModelSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"Model SHA-256 mismatch: expected {options.ExpectedModelSha256}, got {modelSha256}");
        }

        float[] inputValues = ReadFloat32LittleEndian(options.Input, ExpectedInputElements);
        int inputFiniteCount = inputValues.Count(float.IsFinite);
        if (inputFiniteCount != ExpectedInputElements)
        {
            throw new InvalidDataException("Input contains a non-finite FP32 value.");
        }

        var catalog = ExecutionProviderCatalog.GetDefault();
        await catalog.RegisterCertifiedAsync();
        OrtEnv ortEnv = OrtEnv.Instance();
        IReadOnlyList<OrtEpDevice> availableDevices = ortEnv.GetEpDevices();
        if (availableDevices.Count == 0)
        {
            throw new InvalidOperationException("Windows ML ONNX Runtime reported no execution-provider devices.");
        }

        OrtEpDevice selectedDevice = availableDevices
            .OrderBy(device => DevicePriority(device.HardwareDevice.Type))
            .ThenBy(device => device.EpName, StringComparer.Ordinal)
            .First();

        string profilePrefix = Path.Combine(Path.GetTempPath(), $"rimeflow-winml-{Environment.ProcessId}-");
        string? profilePath = null;
        using var sessionOptions = new SessionOptions
        {
            ProfileOutputPathPrefix = profilePrefix,
            EnableProfiling = true,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        sessionOptions.AppendExecutionProvider(ortEnv, [selectedDevice], new Dictionary<string, string>());

        using var session = new InferenceSession(options.Model, sessionOptions);
        ValidateMetadata(session);
        NodeMetadata inputMetadata = session.InputMetadata["images"];
        NodeMetadata outputMetadata = session.OutputMetadata["output0"];
        int[] runtimeInputShape = inputMetadata.Dimensions.ToArray();
        int[] runtimeOutputShape = outputMetadata.Dimensions.ToArray();
        string runtimeInputDtype = DtypeName(inputMetadata.ElementDataType);
        string runtimeOutputDtype = DtypeName(outputMetadata.ElementDataType);

        var inputTensor = new DenseTensor<float>(inputValues, ExpectedInputShape);
        NamedOnnxValue input = NamedOnnxValue.CreateFromTensor("images", inputTensor);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run([input]);
        if (results.Count != 1)
        {
            throw new InvalidDataException($"Expected one runtime output, got {results.Count}.");
        }

        DisposableNamedOnnxValue output = results.Single();
        if (!output.Name.Equals("output0", StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Expected output0, got {output.Name}.");
        }
        float[] outputValues = output.AsEnumerable<float>().ToArray();
        int outputFiniteCount = outputValues.Count(float.IsFinite);
        if (outputValues.Length != ExpectedOutputElements || outputFiniteCount != ExpectedOutputElements)
        {
            throw new InvalidDataException($"Output element/finite count mismatch: {outputValues.Length}/{outputFiniteCount}.");
        }

        IReadOnlyList<OrtEpDevice?> inputEpDevices = session.GetEpDeviceForInputs();
        if (inputEpDevices.Count != 1 || inputEpDevices[0] is null)
        {
            throw new InvalidDataException("Session did not expose the actual EP device for images.");
        }

        profilePath = session.EndProfiling();
        string[] profileProviders = ReadProfileProviders(profilePath);
        if (profileProviders.Length == 0)
        {
            throw new InvalidDataException("ORT profiling did not expose any node execution provider.");
        }

        WriteFloat32LittleEndian(options.Output, outputValues);
        var loadedModules = LoadedWindowsMlModules();
        if (!loadedModules.Any(item => item.Name.Equals("onnxruntime.dll", StringComparison.OrdinalIgnoreCase)) ||
            !loadedModules.Any(item => item.Name.Equals("Microsoft.Windows.AI.MachineLearning.dll", StringComparison.OrdinalIgnoreCase)))
        {
            throw new InvalidOperationException("Actual loaded Windows ML/ONNX Runtime modules could not be identified.");
        }

        var dependencies = DependencyContextPackages();
        if (!dependencies.TryGetValue("Microsoft.WindowsAppSDK.ML", out string? packageVersion) || packageVersion != "2.1.74")
        {
            throw new InvalidOperationException("Runtime dependency context does not contain Microsoft.WindowsAppSDK.ML/2.1.74.");
        }
        if (!dependencies.TryGetValue("Microsoft.Windows.AI.MachineLearning", out string? runtimePackageVersion) || runtimePackageVersion != "2.1.74")
        {
            throw new InvalidOperationException("Runtime dependency context does not contain Microsoft.Windows.AI.MachineLearning/2.1.74.");
        }
        string dotnetSdkVersion = InstalledDotNetSdkVersion();

        WriteJson(options.Report, new
        {
            schemaVersion = 1,
            state = "runtime-verified",
            target,
            runtimeExecuted = true,
            runtimeIntrospectionComplete = true,
            windowsMlApiCalled = true,
            model = new { path = Path.GetFullPath(options.Model), bytes = new FileInfo(options.Model).Length, sha256 = modelSha256, noConversion = true },
            input = new { name = session.InputMetadata.Keys.Single(), count = session.InputMetadata.Count, shape = runtimeInputShape, layout = "NCHW", dtype = runtimeInputDtype, elementCount = inputValues.Length, finiteCount = inputFiniteCount, byteOrder = "little-endian" },
            output = new { name = session.OutputMetadata.Keys.Single(), count = session.OutputMetadata.Count, shape = runtimeOutputShape, layout = "N_ATTRIBUTES_ANCHORS", dtype = runtimeOutputDtype, elementCount = outputValues.Length, finiteCount = outputFiniteCount, byteOrder = "little-endian", path = Path.GetFullPath(options.Output), bytes = new FileInfo(options.Output).Length, sha256 = Sha256File(options.Output) },
            execution = new
            {
                availableDevices = availableDevices.Select(DeviceSnapshot).ToArray(),
                selectedDevice = DeviceSnapshot(selectedDevice),
                sessionInputDevices = inputEpDevices.Select(device => DeviceSnapshot(device!)).ToArray(),
                profileProviders,
                claim = "actual runtime EP/device introspection; profiling may show CPU partitioning and does not imply full-graph acceleration",
            },
            runtime = new
            {
                sourcePackage = new { id = "Microsoft.WindowsAppSDK.ML", version = packageVersion },
                runtimePackage = new { id = "Microsoft.Windows.AI.MachineLearning", version = runtimePackageVersion },
                ortVersion = ortEnv.GetVersionString(),
                dotnetRuntimeVersion = Environment.Version.ToString(),
                sdk = new { version = dotnetSdkVersion, source = "dotnet --version executed at runtime from the publish directory containing global.json" },
                dependencies,
                windowsMlProjection = AssemblySnapshot(typeof(ExecutionProviderCatalog).Assembly),
                onnxRuntimeManaged = AssemblySnapshot(typeof(InferenceSession).Assembly),
                loadedModules = loadedModules.Select(module => new
                {
                    name = module.Name,
                    path = module.Path,
                    fileVersion = module.FileVersion,
                    productVersion = module.ProductVersion,
                    sha256 = module.Sha256,
                }).ToArray(),
            },
            host = HostSnapshot(),
            postprocess = new
            {
                owner = "operator",
                nmsFused = false,
                platformSpecificImplementationAdded = false,
                rawGoldenCommand = $"cargo run --offline --manifest-path evidence/tooling/raw-golden/Cargo.toml -- {options.Output} 768 900 .evidence/windows-ml/decoded/{target}-single-target.json",
            },
        });

        if (profilePath is not null && File.Exists(profilePath))
        {
            File.Delete(profilePath);
        }
    }

    private static void ValidateMetadata(InferenceSession session)
    {
        if (session.InputMetadata.Count != 1 || session.OutputMetadata.Count != 1)
        {
            throw new InvalidDataException($"Expected one input and one output, got {session.InputMetadata.Count}/{session.OutputMetadata.Count}.");
        }
        if (!session.InputMetadata.TryGetValue("images", out NodeMetadata? input) || !input.IsTensor || input.ElementDataType != TensorElementType.Float || !input.Dimensions.SequenceEqual(ExpectedInputShape))
        {
            throw new InvalidDataException("Runtime input metadata drift: expected images FP32 [1,3,640,640].");
        }
        if (!session.OutputMetadata.TryGetValue("output0", out NodeMetadata? output) || !output.IsTensor || output.ElementDataType != TensorElementType.Float || !output.Dimensions.SequenceEqual(ExpectedOutputShape))
        {
            throw new InvalidDataException("Runtime output metadata drift: expected output0 FP32 [1,84,8400].");
        }
    }

    private static int DevicePriority(OrtHardwareDeviceType type) => type switch
    {
        OrtHardwareDeviceType.NPU => 0,
        OrtHardwareDeviceType.GPU => 1,
        OrtHardwareDeviceType.CPU => 2,
        _ => 3,
    };

    private static string DtypeName(TensorElementType type) => type switch
    {
        TensorElementType.Float => "float32",
        _ => type.ToString(),
    };

    private static object DeviceSnapshot(OrtEpDevice device)
    {
        OrtHardwareDevice hardware = device.HardwareDevice;
        using OrtKeyValuePairs epMetadata = device.EpMetadata;
        using OrtKeyValuePairs hardwareMetadata = hardware.Metadata;
        return new
        {
            epName = device.EpName,
            epVendor = device.EpVendor,
            epMetadata = epMetadata.Entries,
            hardware = new { type = hardware.Type.ToString(), vendor = hardware.Vendor, vendorId = hardware.VendorId, deviceId = hardware.DeviceId, metadata = hardwareMetadata.Entries },
        };
    }

    private static string[] ReadProfileProviders(string profilePath)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(profilePath));
        return document.RootElement.EnumerateArray()
            .Where(item => item.TryGetProperty("args", out JsonElement args) && args.TryGetProperty("provider", out _))
            .Select(item => item.GetProperty("args").GetProperty("provider").GetString())
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Select(value => value!)
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToArray();
    }

    private static float[] ReadFloat32LittleEndian(string path, int expectedElements)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length != expectedElements * sizeof(float))
        {
            throw new InvalidDataException($"Expected {expectedElements * sizeof(float)} input bytes, got {bytes.Length}.");
        }
        var values = new float[expectedElements];
        for (int index = 0; index < values.Length; index++)
        {
            values[index] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(index * sizeof(float), sizeof(float))));
        }
        return values;
    }

    private static void WriteFloat32LittleEndian(string path, IReadOnlyList<float> values)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        byte[] bytes = new byte[values.Count * sizeof(float)];
        for (int index = 0; index < values.Count; index++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(index * sizeof(float), sizeof(float)), BitConverter.SingleToInt32Bits(values[index]));
        }
        File.WriteAllBytes(path, bytes);
    }

    private static Dictionary<string, string> DependencyContextPackages()
    {
        string depsPath = Path.Combine(AppContext.BaseDirectory, $"{Assembly.GetEntryAssembly()!.GetName().Name}.deps.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(depsPath));
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (JsonProperty library in document.RootElement.GetProperty("libraries").EnumerateObject())
        {
            int separator = library.Name.LastIndexOf('/');
            if (separator > 0 && library.Value.TryGetProperty("type", out JsonElement type) && type.GetString() == "package")
            {
                result[library.Name[..separator]] = library.Name[(separator + 1)..];
            }
        }
        return result;
    }

    private static string InstalledDotNetSdkVersion()
    {
        var startInfo = new ProcessStartInfo("dotnet", "--version")
        {
            WorkingDirectory = AppContext.BaseDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        using Process process = Process.Start(startInfo) ?? throw new InvalidOperationException("Could not start dotnet --version for SDK introspection.");
        string output = process.StandardOutput.ReadToEnd().Trim();
        string error = process.StandardError.ReadToEnd().Trim();
        process.WaitForExit();
        if (process.ExitCode != 0 || output != "8.0.423")
        {
            throw new InvalidOperationException($"Expected installed .NET SDK 8.0.423, got exit={process.ExitCode}, stdout={output}, stderr={error}.");
        }
        return output;
    }

    private static object AssemblySnapshot(Assembly assembly)
    {
        string path = assembly.Location;
        FileVersionInfo version = FileVersionInfo.GetVersionInfo(path);
        return new { name = assembly.GetName().Name, assemblyVersion = assembly.GetName().Version?.ToString(), fileVersion = version.FileVersion, productVersion = version.ProductVersion, path, sha256 = Sha256File(path) };
    }

    private sealed record ModuleSnapshot(string Name, string Path, string? FileVersion, string? ProductVersion, string Sha256);

    private static ModuleSnapshot[] LoadedWindowsMlModules()
    {
        string[] names = ["onnxruntime.dll", "Microsoft.Windows.AI.MachineLearning.dll", "DirectML.dll"];
        return Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
            .Where(module => names.Contains(module.ModuleName, StringComparer.OrdinalIgnoreCase))
            .Select(module => new ModuleSnapshot(module.ModuleName, module.FileName, module.FileVersionInfo.FileVersion, module.FileVersionInfo.ProductVersion, Sha256File(module.FileName)))
            .OrderBy(module => module.Name, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static object HostSnapshot() => new
    {
        os = OperatingSystem.IsWindows() ? "windows" : OperatingSystem.IsLinux() ? "linux" : "other",
        osDescription = RuntimeInformation.OSDescription,
        osVersion = Environment.OSVersion.VersionString,
        osArchitecture = RuntimeInformation.OSArchitecture.ToString(),
        processArchitecture = RuntimeInformation.ProcessArchitecture.ToString(),
        framework = RuntimeInformation.FrameworkDescription,
        targetFramework = AppContext.TargetFrameworkName,
        machineName = Environment.MachineName,
        processorIdentifier = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER"),
    };

    private static string Sha256File(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static void WriteJson(string path, object value)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, JsonSerializer.Serialize(value, JsonOptions) + Environment.NewLine);
    }

    private static string? FindOption(string[] args, string name)
    {
        int index = Array.IndexOf(args, name);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }

    private sealed record Options(string Model, string Input, string Output, string Report, string ExpectedModelSha256)
    {
        public static Options Parse(string[] args)
        {
            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            for (int index = 0; index < args.Length; index += 2)
            {
                if (index + 1 >= args.Length || !args[index].StartsWith("--", StringComparison.Ordinal) || values.ContainsKey(args[index]))
                {
                    throw new ArgumentException("Usage: --model <onnx> --input <NCHW FP32> --output <raw FP32> --report <JSON> --expected-model-sha256 <SHA>");
                }
                values.Add(args[index], args[index + 1]);
            }
            string Required(string name) => values.TryGetValue(name, out string? value) && !string.IsNullOrWhiteSpace(value) ? value : throw new ArgumentException($"Missing {name}.");
            string expectedSha = Required("--expected-model-sha256").ToLowerInvariant();
            if (expectedSha.Length != 64 || expectedSha.Any(character => !Uri.IsHexDigit(character)))
            {
                throw new ArgumentException("--expected-model-sha256 must be 64 hexadecimal characters.");
            }
            if (values.Count != 5)
            {
                throw new ArgumentException("Unexpected option.");
            }
            return new Options(Required("--model"), Required("--input"), Required("--output"), Required("--report"), expectedSha);
        }
    }
}
