namespace F1_tyre_strategy.Tyres;

abstract class Tyre
{
    public string Name { get; }
    protected double[] Curve { get; }

    protected Tyre(string name, double a, double b, double c, int totalLaps = 60)
    {
        Name = name;

        Curve = new double[totalLaps + 1];
        for (int lap = 0; lap <= totalLaps; lap++)
        {
            Curve[lap] = c + a * Math.Exp(b * lap);
        }
    }

    public virtual double GetLapTime(int stintLap)
    {
        if (stintLap >= Curve.Length)
            return Curve[^1]; // last element
        return Curve[stintLap];
    }
}
