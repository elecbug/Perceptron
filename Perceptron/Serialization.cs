using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Net.Mime;
using System.Text;
using System.Threading.Tasks;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.Versioning;

namespace Perceptron
{
    public class Serialization
    {
        private double[,] BitmapData;

        [SupportedOSPlatform("windows")]
        [SupportedOSPlatform("linux")]
        public Serialization(string filename) 
        {
            using (FileStream pngStream = new FileStream(filename, FileMode.Open, FileAccess.Read))
            using (Bitmap image = new Bitmap(pngStream))
            {
                BitmapData = new double[image.Width, image.Height];

                for (int x = 0; x < image.Width; x++)
                {
                    for (int y = 0; y < image.Height; y++)
                    {
                        BitmapData[x,y] = (image.GetPixel(x, y).R + image.GetPixel(x, y).G + image.GetPixel(x, y).B) / (double)3 / 255;
                    }
                }
            }
        }

        public Serialization(double[,] bitmapData)
        {
            BitmapData = (double[,])bitmapData.Clone();
        }

        public double[,] AverageFilter(int filterEdge)
        {
            if (filterEdge % 2 == 0)
            {
                throw new ArgumentException("The filterEdge must be odd");
            }

            int half = filterEdge / 2;

            double[,] result = new double[BitmapData.GetLength(0) - 2 * half, BitmapData.GetLength(1) - 2 * half];
            int xx = 0, yy = 0;

            for (int x = half + 1; x < BitmapData.GetLength(0) - half; x++)
            {
                for (int y = half + 1; y < BitmapData.GetLength(1) - half; y++)
                {
                    result[xx, yy] = GetNear(x, y, filterEdge);

                    yy++;
                }

                xx++;
                yy = 0;
            }

            return result;
        }

        private double GetNear(int x, int y, int edge)
        {
            int half = edge / 2;
            double result = 0;

            for (int xx = x - half; xx <= x + half; xx++)
            {
                for (int yy = y - half; yy <= y + half; yy++)
                {
                    result += BitmapData[xx, yy];
                }
            }

            return result / edge / edge;
        }
    }
}
