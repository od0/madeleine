namespace Celeste.Mod.InputTruth;

/// <summary>
/// In-game settings, reachable from Mod Options.
///
/// The wild overlay is OFF by default and must stay that way: it is a
/// calibration instrument, not part of the recording format. A normal session
/// carries exactly the two answer-key regions the builder already knows how to
/// mask; enabling this adds a third, which is declared per session in
/// meta.json rather than assumed by anything downstream.
/// </summary>
public sealed class InputTruthSettings : EverestModuleSettings
{
    [SettingName("Wild-style translucent overlay (calibration only)")]
    [SettingSubText(
        "Draws a translucent, labelled input HUD on the right edge, imitating " +
        "a speedrunner's composited overlay. For decoder calibration against " +
        "engine truth. Leave OFF for normal recording sessions.")]
    public bool WildOverlayEnabled { get; set; }
}
