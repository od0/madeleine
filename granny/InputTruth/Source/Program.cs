using System;
using System.Globalization;

namespace Celeste.Mod.InputTruth;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length > 0 &&
            string.Equals(args[0], "--print-vectors", StringComparison.Ordinal))
        {
            return PrintVectors(args);
        }

        if (args.Length == 1 &&
            string.Equals(args[0], "--print-overlay-cells", StringComparison.Ordinal))
        {
            PrintOverlayCells();
            return 0;
        }

        Console.Error.WriteLine(
            "Usage: dotnet run --project granny/InputTruth -- --print-vectors N1 N2 ...");
        Console.Error.WriteLine(
            "   or: dotnet run --project granny/InputTruth -- --print-overlay-cells");
        return 2;
    }

    private static int PrintVectors(string[] args)
    {
        for (int i = 1; i < args.Length; i++)
        {
            if (!int.TryParse(
                    args[i],
                    NumberStyles.None,
                    CultureInfo.InvariantCulture,
                    out int idx) ||
                idx < 0 ||
                idx > FrameIndexStrip.PayloadMask)
            {
                Console.Error.WriteLine(
                    $"Frame index must be an integer from 0 through {FrameIndexStrip.PayloadMask}: {args[i]}");
                return 2;
            }

            Console.Write(idx.ToString(CultureInfo.InvariantCulture));
            Console.Write('\t');
            Console.WriteLine(FrameIndexStrip.Pattern(idx));
        }

        return 0;
    }

    private static void PrintOverlayCells()
    {
        Console.Write('[');
        for (int cell = 0; cell < InputOverlayGeometry.CellCount; cell++)
        {
            if (cell > 0)
                Console.Write(',');

            Console.Write("{\"key\":\"");
            Console.Write(KeyName(cell));
            Console.Write("\",\"rect\":[");
            Console.Write(InputOverlayGeometry.CellX(cell).ToString(CultureInfo.InvariantCulture));
            Console.Write(',');
            Console.Write(InputOverlayGeometry.CellY(cell).ToString(CultureInfo.InvariantCulture));
            Console.Write(',');
            Console.Write(InputOverlayGeometry.CellWidth.ToString(CultureInfo.InvariantCulture));
            Console.Write(',');
            Console.Write(InputOverlayGeometry.CellHeight.ToString(CultureInfo.InvariantCulture));
            Console.Write("]}");
        }
        Console.WriteLine(']');
    }

    private static string KeyName(int cell)
    {
        return cell switch
        {
            0 => "left",
            1 => "right",
            2 => "up",
            3 => "down",
            4 => "jump",
            5 => "dash",
            6 => "grab",
            _ => throw new ArgumentOutOfRangeException(nameof(cell)),
        };
    }
}
