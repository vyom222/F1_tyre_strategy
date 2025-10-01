namespace F1_tyre_strategy.Tyres;

class HardTyre : Tyre
{
    public HardTyre(double a, double b, double c, int totalLaps = 60)
        : base("Hard", a, b, c+2, totalLaps) { }
}
