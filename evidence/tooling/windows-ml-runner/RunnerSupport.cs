using System.Text.Json;

internal enum FailureStage
{
    ArgumentValidation,
    PlatformValidation,
    ModelIdentity,
    InputValidation,
    CatalogRegistration,
    DeviceSelection,
    SessionCreation,
    MetadataValidation,
    Inference,
    OutputValidation,
    ProviderIntrospection,
    ModuleIdentity,
    DependencyIdentity,
    SdkRuntimeIdentity,
    ArtifactPublication,
}

internal sealed class RunState
{
    public FailureStage FailureStage { get; set; } = FailureStage.ArgumentValidation;
    public bool WindowsMlApiCalled { get; set; }
    public bool CatalogRegistrationAttempted { get; set; }
    public bool CatalogRegistrationCompleted { get; set; }
    public bool SessionCreated { get; set; }
    public bool InferenceExecuted { get; set; }
    public bool RuntimeIntrospectionComplete { get; set; }
    public bool OutputPublished { get; set; }

    public void BeginCatalogApiCall()
    {
        FailureStage = FailureStage.CatalogRegistration;
        WindowsMlApiCalled = true;
    }

    public void BeginCatalogRegistration() => CatalogRegistrationAttempted = true;

    public void CompleteCatalogRegistration() => CatalogRegistrationCompleted = true;

    public void MarkSessionCreated() => SessionCreated = true;

    public void MarkInferenceExecuted() => InferenceExecuted = true;

    public void RunStage(FailureStage stage, Action action)
    {
        FailureStage = stage;
        action();
    }

    public object FailureSnapshot(Exception error) => new
    {
        schemaVersion = 1,
        state = "failed",
        runtimeExecuted = WindowsMlApiCalled,
        failureStage = StageName(FailureStage),
        windowsMlApiCalled = WindowsMlApiCalled,
        catalogRegistrationAttempted = CatalogRegistrationAttempted,
        catalogRegistrationCompleted = CatalogRegistrationCompleted,
        sessionCreated = SessionCreated,
        inferenceExecuted = InferenceExecuted,
        runtimeIntrospectionComplete = RuntimeIntrospectionComplete,
        outputPublished = false,
        stages = Snapshot(),
        error = new { type = error.GetType().FullName, message = error.Message },
    };

    public object Snapshot() => new
    {
        failureStage = StageName(FailureStage),
        windowsMlApiCalled = WindowsMlApiCalled,
        catalogRegistrationAttempted = CatalogRegistrationAttempted,
        catalogRegistrationCompleted = CatalogRegistrationCompleted,
        sessionCreated = SessionCreated,
        inferenceExecuted = InferenceExecuted,
        runtimeIntrospectionComplete = RuntimeIntrospectionComplete,
        outputPublished = OutputPublished,
    };

    public static string StageName(FailureStage stage) => stage switch
    {
        FailureStage.ArgumentValidation => "argument-validation",
        FailureStage.PlatformValidation => "platform-validation",
        FailureStage.ModelIdentity => "model-identity",
        FailureStage.InputValidation => "input-validation",
        FailureStage.CatalogRegistration => "catalog-registration",
        FailureStage.DeviceSelection => "device-selection",
        FailureStage.SessionCreation => "session-creation",
        FailureStage.MetadataValidation => "metadata-validation",
        FailureStage.Inference => "inference",
        FailureStage.OutputValidation => "output-validation",
        FailureStage.ProviderIntrospection => "provider-introspection",
        FailureStage.ModuleIdentity => "module-identity",
        FailureStage.DependencyIdentity => "dependency-identity",
        FailureStage.SdkRuntimeIdentity => "sdk-runtime-identity",
        FailureStage.ArtifactPublication => "artifact-publication",
        _ => throw new ArgumentOutOfRangeException(nameof(stage)),
    };
}

internal static class AtomicArtifacts
{
    public static string CreateStagingPath(string destination)
    {
        string fullPath = Path.GetFullPath(destination);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        return Path.Combine(Path.GetDirectoryName(fullPath)!, $".{Path.GetFileName(fullPath)}.{Guid.NewGuid():N}.staging");
    }

    public static void WriteJsonStaging(string stagingPath, object value, JsonSerializerOptions options)
    {
        File.WriteAllText(stagingPath, JsonSerializer.Serialize(value, options) + Environment.NewLine);
    }

    public static void PublishPair(string outputStaging, string outputDestination, string reportStaging, string reportDestination, Action<string>? beforeMove = null)
    {
        string outputBackup = CreateBackupPath(outputDestination);
        string reportBackup = CreateBackupPath(reportDestination);
        bool outputBackedUp = false;
        bool reportBackedUp = false;
        bool outputMoved = false;
        bool reportMoved = false;
        try
        {
            beforeMove?.Invoke("output");
            outputBackedUp = AtomicReplace(outputStaging, outputDestination, outputBackup);
            outputMoved = true;
            beforeMove?.Invoke("report");
            reportBackedUp = AtomicReplace(reportStaging, reportDestination, reportBackup);
            reportMoved = true;
        }
        catch
        {
            if (reportMoved) RestoreBackup(reportDestination, reportBackup, reportBackedUp);
            if (outputMoved) RestoreBackup(outputDestination, outputBackup, outputBackedUp);
            throw;
        }
        finally
        {
            TryDelete(outputStaging);
            TryDelete(reportStaging);
            TryDelete(outputBackup);
            TryDelete(reportBackup);
        }
    }

    public static bool PublishFailureReportIfAbsent(string reportStaging, string reportDestination)
    {
        try
        {
            if (File.Exists(reportDestination)) return false;
            File.Move(reportStaging, reportDestination);
            return true;
        }
        finally
        {
            TryDelete(reportStaging);
        }
    }

    public static void CleanupProfile(string? profilePath, string? profilePrefix)
    {
        if (!string.IsNullOrWhiteSpace(profilePath)) TryDelete(profilePath);
        if (string.IsNullOrWhiteSpace(profilePrefix)) return;
        string directory = Path.GetDirectoryName(profilePrefix)!;
        if (!Directory.Exists(directory)) return;
        foreach (string path in Directory.EnumerateFiles(directory, $"{Path.GetFileName(profilePrefix)}*")) TryDelete(path);
    }

    private static string CreateBackupPath(string destination) => Path.Combine(Path.GetDirectoryName(Path.GetFullPath(destination))!, $".{Path.GetFileName(destination)}.{Guid.NewGuid():N}.backup");

    private static bool AtomicReplace(string staging, string destination, string backup)
    {
        if (File.Exists(destination))
        {
            File.Replace(staging, destination, backup, ignoreMetadataErrors: true);
            return true;
        }
        File.Move(staging, destination);
        return false;
    }

    private static void RestoreBackup(string destination, string backup, bool backedUp)
    {
        if (backedUp)
        {
            File.Replace(backup, destination, null, ignoreMetadataErrors: true);
        }
        else
        {
            DeleteIfExists(destination);
        }
    }

    private static void DeleteIfExists(string path)
    {
        if (File.Exists(path)) File.Delete(path);
    }

    private static void TryDelete(string path)
    {
        try { DeleteIfExists(path); }
        catch { }
    }
}
