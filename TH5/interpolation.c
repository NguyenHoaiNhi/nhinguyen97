float shAMG_PUB_TMP_ConvTemperature(unsigned char aucRegVal[2])
{
	static float temple = 0;
	short shVal = ((short)(aucRegVal[1] & 0x07) << 8) | aucRegVal[0];

	if (0 != (0x08 & aucRegVal[1]))
	{
		shVal -= 2048;
	}

	temple = (float)shVal*0.25;
	return(temple);
}
void vAMG_PUB_TMP_ConvTemperature64(unsigned char* pucRegVal, float* pshVal)
{
	short  ucCnt = 0;
	for (ucCnt = 0; ucCnt < 12769; ucCnt++)
	{
		pshVal[ucCnt] = shAMG_PUB_TMP_ConvTemperature(pucRegVal + ucCnt * 2);
	}
	
}
float* bAMG_PUB_IMG_LinearInterpolation(unsigned char ucWidth, unsigned char ucHeight, short* pshInImg)
{	
	static float c[12769];
	static short b[12769];
	for (int i = 0; i < 12769; i++) {
		b[i] = 0;
	}
	short * pshOutImg = (short*)b;
	const unsigned char c_ucRes = 16;
	int bRet = 0;

	if (pshInImg != pshOutImg)
	{
		unsigned short	usImg = 0;

		for (usImg = 0; usImg < ucWidth * ucHeight; usImg++)
		{
			unsigned char	ucImgX = usImg % ucWidth;
			unsigned char	ucImgY = usImg / ucWidth;
			unsigned short	usSnrXn = (unsigned short)(c_ucRes * ucImgX * (8 - 1) / (ucWidth - 1));
			unsigned short	usSnrYn = (unsigned short)(c_ucRes * ucImgY * (8 - 1) / (ucHeight - 1));
			unsigned char	ucSnrX = (unsigned char)(usSnrXn / c_ucRes);
			unsigned char	ucSnrY = (unsigned char)(usSnrYn / c_ucRes);
			unsigned char	ucRateX1 = (unsigned char)(usSnrXn % c_ucRes);
			unsigned char	ucRateY1 = (unsigned char)(usSnrYn % c_ucRes);
			unsigned char	ucRateX0 = c_ucRes - ucRateX1;
			unsigned char	ucRateY0 = c_ucRes - ucRateY1;
			unsigned char	ucSnr = ucSnrX + ucSnrY * 8;
			long	loWork = 0;

			if (ucImgX == (ucWidth - 1))
			{
				if (ucImgY == (ucHeight - 1))
				{
					loWork += (long)pshInImg[ucSnr];
				}
				else
				{
					loWork += (long)pshInImg[ucSnr] * ucRateY0;
					loWork += (long)pshInImg[ucSnr + 8] * ucRateY1;
					loWork /= c_ucRes;
				}
			}
			else
			{
				if (ucImgY == (ucHeight - 1))
				{
					loWork += (long)pshInImg[ucSnr] * ucRateX0;
					loWork += (long)pshInImg[ucSnr + 1] * ucRateX1;
					loWork /= c_ucRes;
				}
				else
				{
					loWork += (long)pshInImg[ucSnr] * ucRateX0 * ucRateY0;
					loWork += (long)pshInImg[ucSnr + 1] * ucRateX1 * ucRateY0;
					loWork += (long)pshInImg[ucSnr + 8] * ucRateX0 * ucRateY1;
					loWork += (long)pshInImg[ucSnr + 1 + 8] * ucRateX1 * ucRateY1;
					loWork /= c_ucRes * c_ucRes;
				}
			}
			pshOutImg[usImg] = (short)loWork;
		}
		vAMG_PUB_TMP_ConvTemperature64((unsigned char *)b, c);
		bRet = 1;
	}

	return(c);
}
