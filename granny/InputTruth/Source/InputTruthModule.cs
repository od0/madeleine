using System;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Reflection.Emit;
using System.Text.Json;
using Microsoft.Xna.Framework;
using Monocle;

namespace Celeste.Mod.InputTruth;

public sealed class InputTruthModule : EverestModule
{
    public const string ModVersion = "0.2.0";

    // Manifest mask source of truth, in Celeste's 1920x1080 logical
    // game-window coordinates. Device pixels are these values multiplied by
    // the backing scale used by Engine.ScreenMatrix.
    public const int FrameIndexMaskX = 0;
    public const int FrameIndexMaskY = 0;
    public const int FrameIndexCellSize = 16;
    public const int FrameIndexCellCount = FrameIndexStrip.PatternLength;
    public const int FrameIndexQuietZone = FrameIndexCellSize;
    public const int FrameIndexMaskWidth =
        (FrameIndexCellCount + 2) * FrameIndexCellSize;
    public const int FrameIndexMaskHeight = 3 * FrameIndexCellSize;

    public const int InputOverlayMaskX = 0;
    public const int InputOverlayMaskY = 1032;
    public const int InputOverlayMaskWidth = 416;
    public const int InputOverlayMaskHeight = 48;

    public override Type SettingsType => typeof(InputTruthSettings);

    public static InputTruthSettings Settings =>
        (InputTruthSettings?)Instance?._Settings ?? new InputTruthSettings();

    public static InputTruthModule? Instance { get; private set; }

    private static readonly Func<Player, bool> GetPlayerOnGround =
        CreatePlayerOnGroundGetter();

    private long _frameIndex;
    private string _renderPattern = string.Empty;
    private InputFrameState _capturedFrame;
    private TruthCsvWriter? _writer;
    private DateTimeOffset _startedAt;
    private string _sessionId = string.Empty;
    private string _sessionDirectory = string.Empty;
    private string _cachedRoomId = string.Empty;
    private bool _metaWritten;
    private bool _deathTriggered;
    // meta.json is written at frame 0, but the wild-overlay toggle lives in
    // Mod Options and can be flipped after the game has loaded — so a
    // start-of-session snapshot can be STALE, and a stale "false" would let
    // the assembler skip masking a visible answer key. Track the value seen at
    // meta-write time, rewrite meta on flush, and record any mid-session
    // change so downstream can refuse rather than guess.
    private bool _wildOverlayAtMetaWrite;
    private bool _wildOverlayToggledMidSession;
    private DateTimeOffset _frame0WallTime;

    public InputTruthModule()
    {
        Instance = this;
    }

    public override void Load()
    {
        _frameIndex = 0;
        _renderPattern = FrameIndexStrip.Pattern(0);
        _capturedFrame = default;
        _cachedRoomId = string.Empty;
        _deathTriggered = false;
        _metaWritten = false;
        _startedAt = DateTimeOffset.UtcNow;
        _sessionId = "rec_" + _startedAt.UtcDateTime.ToString(
            "yyyyMMdd_HHmmss",
            CultureInfo.InvariantCulture);
        _sessionDirectory = Path.Combine(ResolveOutputRoot(), _sessionId);

        Directory.CreateDirectory(_sessionDirectory);
        _writer = new TruthCsvWriter(
            Path.Combine(_sessionDirectory, "truth_raw.csv"));

        On.Monocle.Engine.Update += OnEngineUpdate;
        On.Monocle.Engine.RenderCore += OnEngineRenderCore;
        Everest.Events.Player.OnDie += OnPlayerDie;
        Everest.Events.Level.OnExit += OnLevelExit;
        Everest.Events.Celeste.OnExiting += OnAppExiting;
    }

    public override void Unload()
    {
        On.Monocle.Engine.Update -= OnEngineUpdate;
        On.Monocle.Engine.RenderCore -= OnEngineRenderCore;
        Everest.Events.Player.OnDie -= OnPlayerDie;
        Everest.Events.Level.OnExit -= OnLevelExit;
        Everest.Events.Celeste.OnExiting -= OnAppExiting;

        DisposeWriter();
    }

    public static Rectangle FrameIndexMaskRect(int backingScale)
    {
        if (backingScale <= 0)
            throw new ArgumentOutOfRangeException(nameof(backingScale));

        return new Rectangle(
            FrameIndexMaskX * backingScale,
            FrameIndexMaskY * backingScale,
            FrameIndexMaskWidth * backingScale,
            FrameIndexMaskHeight * backingScale);
    }

    public static Rectangle InputOverlayMaskRect(int backingScale)
    {
        if (backingScale <= 0)
            throw new ArgumentOutOfRangeException(nameof(backingScale));

        return new Rectangle(
            InputOverlayMaskX * backingScale,
            InputOverlayMaskY * backingScale,
            InputOverlayMaskWidth * backingScale,
            InputOverlayMaskHeight * backingScale);
    }

    private void OnEngineUpdate(
        On.Monocle.Engine.orig_Update orig,
        Engine self,
        GameTime gameTime)
    {
        orig(self, gameTime);

        TruthCsvWriter? writer = _writer;
        if (writer is null)
            return;

        long capturedFrameIndex = _frameIndex;
        _renderPattern = FrameIndexStrip.Pattern(unchecked((int)capturedFrameIndex));

        if (!_metaWritten)
        {
            WriteMeta(DateTimeOffset.UtcNow);
            _metaWritten = true;
        }

        bool death = _deathTriggered;
        _deathTriggered = false;
        _capturedFrame = CaptureFrame(capturedFrameIndex, death);
        writer.WriteRow(in _capturedFrame);

        _frameIndex = capturedFrameIndex + 1;
    }

    private void OnEngineRenderCore(
        On.Monocle.Engine.orig_RenderCore orig,
        Engine self)
    {
        orig(self);
        FrameIndexStrip.Render(_renderPattern);
        InputOverlay.Render(in _capturedFrame);
        // Calibration-only, off by default. Drawn last so it composites over
        // game content the way a streamer's overlay does.
        if (Settings.WildOverlayEnabled)
            WildOverlay.Render(in _capturedFrame);
    }

    private void OnPlayerDie(Player player)
    {
        _deathTriggered = true;
    }

    private void OnLevelExit(
        Level level,
        LevelExit exit,
        LevelExit.Mode mode,
        Session session,
        HiresSnow snow)
    {
        _writer?.Flush();
        // Refresh the declaration: the toggle may have moved since frame 0.
        if (_metaWritten)
            WriteMetaFile();
    }

    private void OnAppExiting()
    {
        DisposeWriter();
    }

    private void DisposeWriter()
    {
        TruthCsvWriter? writer = _writer;
        _writer = null;

        if (writer is null)
            return;

        if (!_metaWritten)
        {
            WriteMeta(_startedAt);
            _metaWritten = true;
        }

        writer.Dispose();
    }

    private void WriteMeta(DateTimeOffset frame0WallTime)
    {
        _frame0WallTime = frame0WallTime;
        _wildOverlayAtMetaWrite = Settings.WildOverlayEnabled;
        WriteMetaFile();
    }

    /// <summary>
    /// Rewrite meta.json from the CURRENT setting state. Called at frame 0 and
    /// again on every flush, so the file a session ships with reflects what was
    /// actually rendered rather than what was configured before the player
    /// reached Mod Options.
    /// </summary>
    private void WriteMetaFile()
    {
        if (Settings.WildOverlayEnabled != _wildOverlayAtMetaWrite)
            _wildOverlayToggledMidSession = true;

        string json = JsonSerializer.Serialize(
            new
            {
                session_id = _sessionId,
                mod_version = ModVersion,
                everest_version = Everest.VersionString,
                celeste_version = Everest.VersionCelesteString,
                overlay_style = "inputtruth-v1",
                // Declared per session rather than assumed downstream: when
                // the calibration overlay is on it is a THIRD answer-key
                // region, and the assembler masks whatever rect is named here.
                wild_overlay = Settings.WildOverlayEnabled,
                wild_overlay_rect_logical = Settings.WildOverlayEnabled
                    ? new[]
                    {
                        WildOverlay.PanelX, WildOverlay.PanelY,
                        WildOverlay.PanelWidth, WildOverlay.PanelHeight,
                    }
                    : null,
                // True if the toggle moved after meta was first written; the
                // assembler refuses such a session rather than masking a
                // region that was only sometimes drawn.
                wild_overlay_toggled_mid_session = _wildOverlayToggledMidSession,
                started_at = _startedAt.ToString("O", CultureInfo.InvariantCulture),
                frame0_wall_time = _frame0WallTime.ToString(
                    "O",
                    CultureInfo.InvariantCulture),
            },
            new JsonSerializerOptions { WriteIndented = true });

        File.WriteAllText(Path.Combine(_sessionDirectory, "meta.json"), json);
    }

    private static string ResolveOutputRoot()
    {
        string? overrideRoot = Environment.GetEnvironmentVariable("INPUTTRUTH_OUT");
        if (!string.IsNullOrWhiteSpace(overrideRoot))
            return overrideRoot;

        string? home = Environment.GetEnvironmentVariable("HOME");
        if (string.IsNullOrWhiteSpace(home))
            home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

        if (string.IsNullOrWhiteSpace(home))
            throw new InvalidOperationException(
                "HOME is not set and the user profile directory is unavailable.");

        return Path.Combine(home, "madeleine_sessions", "inputtruth");
    }

    private InputFrameState CaptureFrame(long frameIndex, bool death)
    {
        int moveX = Input.MoveX.Value;
        int moveY = Input.MoveY.Value;
        bool left = moveX < 0;
        bool right = moveX > 0;
        bool up = moveY < 0;
        bool down = moveY > 0;
        bool jump = Input.Jump.Check;
        bool dash = Input.Dash.Check || Input.CrouchDash.Check;
        bool grab = Input.Grab.Check;

        Level? level = Engine.Scene as Level;
        string roomId = CacheRoomId(level);
        Player? player = level?.Tracker.GetEntity<Player>();

        if (player is null)
        {
            return new InputFrameState(
                frameIndex,
                left,
                right,
                up,
                down,
                jump,
                dash,
                grab,
                inputActive: false,
                roomId,
                positionX: float.NaN,
                positionY: float.NaN,
                speedX: float.NaN,
                speedY: float.NaN,
                dashCount: -1,
                stamina: float.NaN,
                onGround: false,
                death);
        }

        bool inputActive =
            level is not null &&
            !level.Paused &&
            !level.Transitioning &&
            !level.InCutscene &&
            !player.Dead;

        Vector2 position = player.Position;
        Vector2 speed = player.Speed;

        return new InputFrameState(
            frameIndex,
            left,
            right,
            up,
            down,
            jump,
            dash,
            grab,
            inputActive,
            roomId,
            position.X,
            position.Y,
            speed.X,
            speed.Y,
            player.Dashes,
            player.Stamina,
            GetPlayerOnGround(player),
            death);
    }

    private string CacheRoomId(Level? level)
    {
        string roomId = level?.Session?.Level ?? string.Empty;
        if (!string.Equals(roomId, _cachedRoomId, StringComparison.Ordinal))
            _cachedRoomId = roomId;

        return _cachedRoomId;
    }

    private static Func<Player, bool> CreatePlayerOnGroundGetter()
    {
        FieldInfo field = typeof(Player).GetField(
                "onGround",
                BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingFieldException(typeof(Player).FullName, "onGround");

        DynamicMethod method = new(
            "InputTruth_GetPlayerOnGround",
            typeof(bool),
            new[] { typeof(Player) },
            typeof(InputTruthModule),
            skipVisibility: true);

        ILGenerator il = method.GetILGenerator();
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ldfld, field);
        il.Emit(OpCodes.Ret);

        return (Func<Player, bool>)method.CreateDelegate(typeof(Func<Player, bool>));
    }
}
