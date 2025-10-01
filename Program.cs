namespace F1_tyre_strategy;

using System;
using System.Diagnostics;
using System.Text.Json;
using F1_tyre_strategy.Tyres;
using F1_tyre_strategy.Strategy;

class Program
{
    static void Main()
    {
        Console.WriteLine("Fetching tyre curves from Python...");

        string scriptPath = @"/Users/vyomchamaria/Desktop/F1_tyre_strategy/get_curves.py";
        var curves = RunPython(scriptPath);

        if (curves.Count == 0)
        {
            Console.WriteLine("No curves returned from Python, exiting...");
            return;
        }

        // Instantiate tyres with coefficients
        var tyres = new List<Tyre>
        {
            new SoftTyre(
                curves["SOFT"]["a"].GetDouble(),
                curves["SOFT"]["b"].GetDouble(),
                curves["SOFT"]["c"].GetDouble()
            ),
            new MediumTyre(
                curves["MEDIUM"]["a"].GetDouble(),
                curves["MEDIUM"]["b"].GetDouble(),
                curves["MEDIUM"]["c"].GetDouble()
            ),
            new HardTyre(
                curves["HARD"]["a"].GetDouble(),
                curves["HARD"]["b"].GetDouble(),
                curves["HARD"]["c"].GetDouble()
            )
        };

        // Example: 60-lap race, 20s pit loss
        var solver = new RaceStrategySolver(totalLaps: 70, pitLoss: 25.0, tyres: tyres);
        var (bestTime, bestStrategy) = solver.Solve();

        Console.WriteLine($"Best total time: {bestTime:F2}s");
        Console.WriteLine("Strategy:");
        foreach (var stint in bestStrategy)
        {
            Console.WriteLine($"  - {stint}");
        }
    }

    static Dictionary<string, Dictionary<string, JsonElement>> RunPython(string scriptPath)
    {
        var psi = new ProcessStartInfo
        {
            FileName = "/opt/homebrew/bin/python3.11",
            Arguments = scriptPath,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = Process.Start(psi);
        string output = process.StandardOutput.ReadToEnd();
        string errors = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (!string.IsNullOrWhiteSpace(errors))
        {
            Console.WriteLine("Python errors: " + errors);
        }

        try
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            var tyreData = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, JsonElement>>>(output, options);

            Console.WriteLine("Fetched tyre curves from Python...");
            foreach (var tyre in tyreData!)
            {
                string equation = tyre.Value["equation"].GetString();
                Console.WriteLine($"{tyre.Key}: {equation}");
            }

            return tyreData!;
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to parse Python output:");
            Console.WriteLine(output);
            Console.WriteLine(ex);
            return new Dictionary<string, Dictionary<string, JsonElement>>();
        }
    }
}
