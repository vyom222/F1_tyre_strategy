namespace F1_tyre_strategy.Tyres;

class SoftTyre : Tyre
{
    public SoftTyre(double a, double b, double c, int totalLaps = 60)
        : base("Soft", a, b, c, totalLaps) { }
}