using System.Drawing;

Perceptron.Classification classification = new Perceptron.Classification(
        2,
        new int[] { 4, 2 },
        new List<Func<double, double>> { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Sigmoid }
    );

List<double[]> inputs = new List<double[]>()
    {
        new[] { 0.0, 0.0 },
        new[] { 0.0, 1.0 },
        new[] { 1.0, 0.0 },
        new[] { 1.0, 1.0 }
    };
List<double[]> outputs = new List<double[]>()
    {
        new[] { 0.0, 1.0 },
        new[] { 1.0, 0.0 },
        new[] { 1.0, 0.0 },
        new[] { 0.0, 1.0 },
    };

if (File.Exists("xor-classification.json"))
{
    classification.Load("xor-classification.json");
}

Bitmap bitmap = new Bitmap(1000, 1000);

for (int x = 0; x < bitmap.Width; x++)
{
    for (int y = 0; y < bitmap.Height; y++)
    {
        classification.Test(new double[] { x / 1000.0, y / 1000.0 }, out double[] o);
        bitmap.SetPixel(x, y, Color.FromArgb(red: (int)(o[0] * 255), blue: (int)(o[1] * 255), green: 0));
    }
}

bitmap.Save("xor-map.png");

//for (int j = 0; j < 10; j++)
//{
//    classification.Learn(inputs, outputs, 1000, 0.0000001, 0.001, 100);

//    for (int i = 0; i < inputs.Count; i++)
//    {
//        classification.Test(inputs[i], out double[] o);

//        foreach (var x in o)
//        {
//            Console.WriteLine(x.ToString("0.000-000-000-000"));
//        }

//        Console.WriteLine();
//    }

//    classification.Save("xor-classification.json");
//}