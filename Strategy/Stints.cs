using F1_tyre_strategy.Tyres;

namespace F1_tyre_strategy.Strategy;

class Stint
{
    public Tyre Tyre { get; }
    public int Length { get; }
    public double TotalTime { get; }

    public Stint(Tyre tyre, int length)
    {
        Tyre = tyre;
        Length = length;

        double sum = 0;
        for (int i = 0; i < length; i++)
            sum += tyre.GetLapTime(i);

        TotalTime = sum;
    }

    public override string ToString()
    {
        return $"{Length} laps on {Tyre.Name}";
    }
}
