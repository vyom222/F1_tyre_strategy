using F1_tyre_strategy.Tyres;

namespace F1_tyre_strategy.Strategy;

class RaceStrategySolver
{
    private readonly int totalLaps;
    private readonly double pitLoss;
    private readonly List<Tyre> tyres;

    public RaceStrategySolver(int totalLaps, double pitLoss, List<Tyre> tyres)
    {
        this.totalLaps = totalLaps;
        this.pitLoss = pitLoss;
        this.tyres = tyres;
    }

    // Bitmask mapping for compounds
    private int GetCompoundMask(string name) => name.ToUpper() switch
    {
        "SOFT" => 1 << 0,
        "MEDIUM" => 1 << 1,
        "HARD" => 1 << 2,
        _ => 0
    };

    public (double bestTime, List<Stint> bestStrategy) Solve()
    {
        // dp[lap][mask] = (time, strategy)
        var dp = new (double time, List<Stint> strategy)?[totalLaps + 1, 1 << tyres.Count];

        dp[0, 0] = (0, new List<Stint>());

        for (int lap = 0; lap < totalLaps; lap++)
        {
            for (int mask = 0; mask < (1 << tyres.Count); mask++)
            {
                if (dp[lap, mask] == null) continue;

                var (currentTime, currentStrategy) = dp[lap, mask].Value;

                foreach (var tyre in tyres)
                {
                    int tyreMask = GetCompoundMask(tyre.Name);

                    for (int stintLen = 1; stintLen <= totalLaps - lap; stintLen++)
                    {
                        var stint = new Stint(tyre, stintLen);
                        double newTime = currentTime + stint.TotalTime + (lap > 0 ? pitLoss : 0);

                        int newMask = mask | tyreMask;
                        var existing = dp[lap + stintLen, newMask];

                        if (existing == null || newTime < existing.Value.time)
                        {
                            var newStrategy = new List<Stint>(currentStrategy) { stint };
                            dp[lap + stintLen, newMask] = (newTime, newStrategy);
                        }
                    }
                }
            }
        }

        // Pick best among states that used at least 2 different compounds
        double bestTime = double.PositiveInfinity;
        List<Stint> bestStrategy = new();

        for (int mask = 0; mask < (1 << tyres.Count); mask++)
        {
            if (CountBits(mask) < 2) continue; // must use at least 2 compounds

            var result = dp[totalLaps, mask];
            if (result != null && result.Value.time < bestTime)
            {
                bestTime = result.Value.time;
                bestStrategy = result.Value.strategy;
            }
        }

        return (bestTime, bestStrategy);
    }

    private int CountBits(int x)
    {
        int count = 0;
        while (x > 0)
        {
            count += (x & 1);
            x >>= 1;
        }
        return count;
    }
}
