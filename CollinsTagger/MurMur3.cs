/*
This code is public domain.
 
The MurmurHash3 algorithm was created by Austin Appleby and put into the public domain. See http://code.google.com/p/smhasher/
 
This C# variant was authored by
Elliott B. Edwards and was placed into the public domain as a gist
Status...Working on verification (Test Suite)
Set up to run as a LinqPad (linqpad.net) script (thus the ".Dump()" call)
*/

using System.IO;
namespace CollinsTagger
{
    public static class MurMurHash3
    {
        //Change to suit your needs
        const uint seed = 144;

        public static int Hash(byte[] data)
        {
            const uint c1 = 0xcc9e2d51;
            const uint c2 = 0x1b873593;

            uint h1 = seed;
            uint k1 = 0;
            uint streamLength = 0;

            // More efficient approach for small byte arrays - avoid MemoryStream overhead
            if (data == null || data.Length == 0)
            {
                return (int)seed;
            }

            streamLength = (uint)data.Length;
            
            // Process 4 bytes at a time directly from the array
            int pos = 0;
            while (pos <= data.Length - 4)
            {
                // Get four bytes from the input into an uint
                k1 = BitConverter.ToUInt32(data, pos);
                
                // Bitmagic hash
                k1 *= c1;
                k1 = rotl32(k1, 15);
                k1 *= c2;

                h1 ^= k1;
                h1 = rotl32(h1, 13);
                h1 = h1 * 5 + 0xe6546b64;
                
                pos += 4;
            }
            
            // Handle remaining bytes
            if (pos < data.Length)
            {
                k1 = 0;
                int remaining = data.Length - pos;
                
                switch (remaining)
                {
                    case 3:
                        k1 = (uint)(data[pos + 2] << 16 | data[pos + 1] << 8 | data[pos]);
                        break;
                    case 2:
                        k1 = (uint)(data[pos + 1] << 8 | data[pos]);
                        break;
                    case 1:
                        k1 = (uint)data[pos];
                        break;
                }
                
                k1 *= c1;
                k1 = rotl32(k1, 15);
                k1 *= c2;
                h1 ^= k1;
            }
            // finalization, magic chants to wrap it all up
            h1 ^= streamLength;
            h1 = fmix(h1);

            unchecked //ignore overflow
            {
                return (int)h1;
            }
        }

        private static uint rotl32(uint x, byte r)
        {
            return (x << r) | (x >> (32 - r));
        }

        private static uint fmix(uint h)
        {
            h ^= h >> 16;
            h *= 0x85ebca6b;
            h ^= h >> 13;
            h *= 0xc2b2ae35;
            h ^= h >> 16;
            return h;
        }
    }
}


