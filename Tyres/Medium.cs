namespace F1_tyre_strategy.Tyres;

class MediumTyre : Tyre
{
    public MediumTyre(double a, double b, double c, int totalLaps = 60)
        : base("Medium", a, b, c, totalLaps) { }
}
