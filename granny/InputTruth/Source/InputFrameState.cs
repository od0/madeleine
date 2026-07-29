namespace Celeste.Mod.InputTruth;

internal readonly struct InputFrameState
{
    public readonly long FrameIndex;
    public readonly bool Left;
    public readonly bool Right;
    public readonly bool Up;
    public readonly bool Down;
    public readonly bool Jump;
    public readonly bool Dash;
    public readonly bool Grab;
    public readonly bool InputActive;
    public readonly string RoomId;
    public readonly float PositionX;
    public readonly float PositionY;
    public readonly float SpeedX;
    public readonly float SpeedY;
    public readonly int DashCount;
    public readonly float Stamina;
    public readonly bool OnGround;
    public readonly bool Death;

    public InputFrameState(
        long frameIndex,
        bool left,
        bool right,
        bool up,
        bool down,
        bool jump,
        bool dash,
        bool grab,
        bool inputActive,
        string roomId,
        float positionX,
        float positionY,
        float speedX,
        float speedY,
        int dashCount,
        float stamina,
        bool onGround,
        bool death)
    {
        FrameIndex = frameIndex;
        Left = left;
        Right = right;
        Up = up;
        Down = down;
        Jump = jump;
        Dash = dash;
        Grab = grab;
        InputActive = inputActive;
        RoomId = roomId;
        PositionX = positionX;
        PositionY = positionY;
        SpeedX = speedX;
        SpeedY = speedY;
        DashCount = dashCount;
        Stamina = stamina;
        OnGround = onGround;
        Death = death;
    }

    public bool IsKeyDown(int cell)
    {
        return cell switch
        {
            0 => Left,
            1 => Right,
            2 => Up,
            3 => Down,
            4 => Jump,
            5 => Dash,
            6 => Grab,
            _ => false,
        };
    }
}
