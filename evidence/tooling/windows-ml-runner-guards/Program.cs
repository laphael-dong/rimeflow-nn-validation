using System.Text.Json;

internal static class Program
{
    private static readonly List<string> Passed = [];

    public static int Main()
    {
        string root = Path.Combine(Path.GetTempPath(), $"rimeflow-winml-guards-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            StageStateGuards();
            AtomicRollbackGuards(root);
            FailurePublicationGuards(root);
            ProfileCleanupGuard(root);
            Console.WriteLine(JsonSerializer.Serialize(new { ok = true, passed = Passed }));
            return 0;
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void StageStateGuards()
    {
        var catalog = new RunState();
        catalog.BeginCatalogApiCall();
        catalog.BeginCatalogRegistration();
        CaptureFailure(() => throw new InvalidOperationException("injected registration failure"));
        Assert(catalog.WindowsMlApiCalled && catalog.CatalogRegistrationAttempted && !catalog.CatalogRegistrationCompleted, "catalog-called-state-preserved");
        var session = new RunState();
        session.MarkSessionCreated();
        CaptureFailure(() => session.RunStage(FailureStage.MetadataValidation, () => throw new InvalidDataException("injected metadata failure")));
        Assert(session.SessionCreated && session.FailureStage == FailureStage.MetadataValidation, "session-created-state-preserved");
        var inference = new RunState();
        inference.MarkSessionCreated();
        inference.MarkInferenceExecuted();
        CaptureFailure(() => inference.RunStage(FailureStage.ProviderIntrospection, () => throw new InvalidDataException("injected introspection failure")));
        Assert(inference.InferenceExecuted && !inference.RuntimeIntrospectionComplete && inference.FailureStage == FailureStage.ProviderIntrospection, "inference-state-preserved-before-introspection");
    }

    private static void AtomicRollbackGuards(string root)
    {
        foreach (FailureStage stage in new[] { FailureStage.ModuleIdentity, FailureStage.DependencyIdentity, FailureStage.SdkRuntimeIdentity })
        {
            string stageName = RunState.StageName(stage);
            string output = Path.Combine(root, $"{stageName}.raw");
            byte[] original = [0x00, 0x7f, 0x80, 0xff];
            File.WriteAllBytes(output, original);
            var state = new RunState { InferenceExecuted = true };
            Exception failure = CaptureFailure(() => state.RunStage(stage, () => throw new InvalidDataException($"injected {stageName}")));
            using JsonDocument snapshot = JsonDocument.Parse(JsonSerializer.Serialize(state.FailureSnapshot(failure)));
            Assert(state.FailureStage == stage && state.InferenceExecuted && !state.OutputPublished && snapshot.RootElement.GetProperty("outputPublished").GetBoolean() == false && File.ReadAllBytes(output).SequenceEqual(original), $"{stageName}-failure-keeps-output-and-state");
        }

        string outputPath = Path.Combine(root, "output.raw");
        string reportPath = Path.Combine(root, "report.json");
        File.WriteAllText(outputPath, "old-output");
        File.WriteAllText(reportPath, "old-report");
        string outputStage = AtomicArtifacts.CreateStagingPath(outputPath);
        string reportStage = AtomicArtifacts.CreateStagingPath(reportPath);
        File.WriteAllText(outputStage, "new-output");
        File.WriteAllText(reportStage, "new-report");
        AssertThrows(() => AtomicArtifacts.PublishPair(outputStage, outputPath, reportStage, reportPath, move => { if (move == "report") throw new IOException("injected report move failure"); }));
        Assert(File.ReadAllText(outputPath) == "old-output" && File.ReadAllText(reportPath) == "old-report", "pair-publication-rolls-back-existing-files");

        string missingDirectory = Path.Combine(root, "missing", "report.json");
        AssertThrows(() => AtomicArtifacts.WriteJsonStaging(missingDirectory, new { state = "failed" }, new JsonSerializerOptions()));
        Assert(File.ReadAllText(outputPath) == "old-output" && File.ReadAllText(reportPath) == "old-report", "staging-write-failure-preserves-finals");

        outputStage = AtomicArtifacts.CreateStagingPath(outputPath);
        reportStage = AtomicArtifacts.CreateStagingPath(reportPath);
        File.WriteAllText(outputStage, "new-output");
        File.WriteAllText(reportStage, "success-report");
        AssertThrows(() => AtomicArtifacts.PublishPair(outputStage, outputPath, reportStage, reportPath, move => { if (move == "output") throw new IOException("injected output publication failure"); }));
        Assert(File.ReadAllText(reportPath) == "old-report", "no-success-report-after-output-publication-failure");
    }

    private static void FailurePublicationGuards(string root)
    {
        string outputPath = Path.Combine(root, "failure-output.raw");
        string reportPath = Path.Combine(root, "failure-report.json");
        byte[] oldOutput = [1, 2, 3, 4];
        string oldReport = "old-report";
        File.WriteAllBytes(outputPath, oldOutput);
        File.WriteAllText(reportPath, oldReport);

        var state = new RunState { FailureStage = FailureStage.DependencyIdentity, InferenceExecuted = true };
        string sidecar = $"{reportPath}.failed.json";
        string staging = AtomicArtifacts.CreateStagingPath(sidecar);
        AtomicArtifacts.WriteJsonStaging(staging, state.FailureSnapshot(new InvalidDataException("identity")), new JsonSerializerOptions());
        Assert(AtomicArtifacts.PublishFailureReportIfAbsent(staging, sidecar), "failure-report-sidecar-published");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(sidecar));
        Assert(document.RootElement.GetProperty("outputPublished").GetBoolean() == false && File.ReadAllBytes(outputPath).SequenceEqual(oldOutput) && File.ReadAllText(reportPath) == oldReport, "failure-report-preserves-existing-output-and-report");

        string absentOutput = Path.Combine(root, "absent-output.raw");
        Assert(!File.Exists(absentOutput), "failure-before-publication-does-not-create-output");
    }

    private static void ProfileCleanupGuard(string root)
    {
        string prefix = Path.Combine(root, "profile-");
        string first = prefix + "one.json";
        string second = prefix + "two.json";
        File.WriteAllText(first, "profile");
        File.WriteAllText(second, "profile");
        try { throw new IOException("injected introspection failure"); }
        catch (IOException) { }
        finally { AtomicArtifacts.CleanupProfile(first, prefix); }
        Assert(!File.Exists(first) && !File.Exists(second), "profile-finally-cleanup");
    }

    private static void Assert(bool condition, string name)
    {
        if (!condition) throw new InvalidOperationException(name);
        Passed.Add(name);
    }

    private static void AssertThrows(Action action)
    {
        try { action(); }
        catch { return; }
        throw new InvalidOperationException("expected exception");
    }

    private static Exception CaptureFailure(Action action)
    {
        try { action(); }
        catch (Exception error) { return error; }
        throw new InvalidOperationException("expected exception");
    }
}
