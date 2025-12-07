//#define _SILENCE
//#define CONVNET_NAME "LIV.IC.UNICAMP.MobILive.best.pf83.norm.convnet"             // 4.0% err
//#define CONVNET_NAME "LIV.IC.UNICAMP.MobILive.best.pf83.nonorm.convnet"           // 4.0% e44
//#define CONVNET_NAME "LIV.IC.UNICAMP.MobILive.best.screening.convnet"             // 1.6% err3

// 1st submission
//#define CONVNET_NAME "LIV.IC.UNICAMP.MobILive.best.screening.handtunning.convnet" // 0.5% err => 1st submission
//#define NORM_NAME    "LIV.IC.UNICAMP.MobILive.norm"
//#define MODEL_NAME   "LIV.IC.UNICAMP.MobILive.model"


//// 2nd submission
//#define CONVNET_NAME "LIV.IC.UNICAMP.MobILive.2nd.convnet"
//#define NORM_NAME    "LIV.IC.UNICAMP.MobILive.2nd.norm"
//#define MODEL_NAME   "LIV.IC.UNICAMP.MobILive.2nd.model"


// 3nd submission
#define nCNNs 1
enum { COMB_SUM, COMB_MAX, COMB_MV }; /* Combination rule type  enumeration */
#define COMB_RULE COMB_SUM
//#define COMB_RULE COMB_MV

#define CONVNET_NAME "LIV.IC.UNICAMP.MobILive.5th.#.convnet"
#define NORM_NAME    "LIV.IC.UNICAMP.MobILive.5th.#.norm"
#define MODEL_NAME   "LIV.IC.UNICAMP.MobILive.5th.#.model"

//#define nCNNs 1
//enum { COMB_SUM, COMB_MAX, COMB_MV }; /* Combination rule type  enumeration */
//#define COMB_RULE COMB_SUM
//
//#define CONVNET_NAME "LIV.IC.UNICAMP.MobILive.test.#.convnet"
//#define NORM_NAME    "LIV.IC.UNICAMP.MobILive.test.#.norm"
//#define MODEL_NAME   "LIV.IC.UNICAMP.MobILive.test.#.model"

char *replace_str(const char *str, const char *old, const char *new)
{
	char *ret, *r;
	const char *p, *q;
	size_t oldlen = strlen(old), newlen = strlen(new);
	size_t count, retlen;

	if (oldlen != newlen) {
		for (count = 0, p = str; (q = strstr(p, old)) != NULL; p = q + oldlen)
			count++;
		/* this is undefined if p - str > PTRDIFF_MAX */
		retlen = p - str + strlen(p) + count * (newlen - oldlen);
	} else
		retlen = strlen(str);

	if ((ret = malloc(retlen + 1)) == NULL)
		return NULL;

	for (r = ret, p = str; (q = strstr(p, old)) != NULL; p = q + oldlen) {
		/* this is undefined if q - p > PTRDIFF_MAX */
		ptrdiff_t l = q - p;
		memcpy(r, p, l);
		r += l;
		memcpy(r, new, newlen);
		r += newlen;
	}
	strcpy(r, p);

	return ret;
}
