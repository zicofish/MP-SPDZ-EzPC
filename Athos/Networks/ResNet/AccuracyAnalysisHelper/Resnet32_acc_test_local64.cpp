#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include<fstream>
#include<Eigen/Dense>
#include<cstdlib>
#include<string>

using namespace std;
using namespace Eigen;

uint32_t public_lrshift(uint32_t x, uint32_t y){
return (x >> y);
}

int32_t public_lrshift(int32_t x, uint32_t y){
return ((int32_t)(((uint32_t)x) >> y));
}

template<typename T>
vector<T> make_vector(size_t size) {
return std::vector<T>(size);
}

template <typename T, typename... Args>
auto make_vector(size_t first, Args... sizes)
{
auto inner = make_vector<T>(sizes...);
return vector<decltype(inner)>(first, inner);
}

template<typename T>
ostream& operator<< (ostream &os, const vector<T> &v)
{
for(auto it = v.begin (); it != v.end (); ++it) {
os << *it << endl;
}
return os;
}


uint32_t signedgtbl(int32_t x, int32_t y){

uint32_t ux = x;

uint32_t uy = y;

uint32_t signBitX = (x & ( (int32_t)1 <<  (int32_t)31));

uint32_t signBitY = (y & ( (int32_t)1 <<  (int32_t)31));
return ((signBitX ^ signBitY) >  (uint32_t)0) ? (signBitX >  (uint32_t)0) ? 0 : 1 : (ux > uy);
}

int32_t signedarshiftbl(int32_t x, uint32_t y){

uint32_t ux = x;

uint32_t signBitX = (x & ( (int32_t)1 <<  (int32_t)31));
return (signBitX >  (uint32_t)0) ? ( (uint32_t)0 - (( (uint32_t)0 - ux) >> y)) : (ux >> y);
}

uint32_t unsignedltbl(uint32_t x, uint32_t y){
return (y > x);
}

uint32_t signedltbl(int32_t x, int32_t y){
return (y > x);
}

uint32_t unsignedleqbl(uint32_t x, uint32_t y){
return ! (x > y);
}

uint32_t signedleqbl(int32_t x, int32_t y){
return ! (x > y);
}

uint32_t unsignedgeqbl(uint32_t x, uint32_t y){
return ! (y > x);
}

uint32_t signedgeqbl(int32_t x, int32_t y){
return ! (y > x);
}

uint32_t unsignedequalsbl(uint32_t x, uint32_t y){
return (! (x < y) && ! (y < x));
}

uint32_t signedequalsbl(int32_t x, int32_t y){
return (! (x < y) && ! (y < x));
}

uint32_t longDivision(uint32_t x, uint32_t y, uint32_t getQuotient){

uint32_t q =  (uint32_t)0;

uint32_t divisor =  (uint32_t)0;

uint32_t cond = 0;
for (uint32_t iter =  (int32_t)0; iter <  (int32_t)32; iter++){

uint32_t i = ( (int32_t)31 - iter);
divisor = (divisor <<  (uint32_t)1);
divisor = (divisor + (public_lrshift((x & ( (uint32_t)1 << i)), i)));
cond = (divisor >= y);
divisor = cond ? (divisor - y) : divisor;
q = (q <<  (uint32_t)1);
q = cond ? (q +  (uint32_t)1) : q;
}
return getQuotient ? q : divisor;
}

uint32_t unsigneddivbl(uint32_t x, uint32_t y){
return longDivision(x, y, 1);
}

uint32_t unsigneddival(uint32_t x, uint32_t y){

uint32_t bx = x;

uint32_t by = y;
return (bx / by);
}

int32_t signeddivbl(int32_t x, int32_t y){

uint32_t isXNeg = (x <  (int32_t)0);

uint32_t isYNeg = (y <  (int32_t)0);

uint32_t ux = isXNeg ? ( (int32_t)0 - x) : x;

uint32_t uy = isYNeg ? ( (int32_t)0 - y) : y;

uint32_t ures = (ux / uy);

uint32_t isResNeg = (isXNeg ^ isYNeg);
return isResNeg ? ( (uint32_t)0 - ures) : ures;
}

int32_t signeddival(int32_t x, int32_t y){

int32_t bx = x;

int32_t by = y;
return (bx / by);
}

uint32_t unsignedmodbl(uint32_t x, uint32_t y){
return longDivision(x, y, 0);
}

uint32_t unsignedmodal(uint32_t x, uint32_t y){

uint32_t bx = x;

uint32_t by = y;
return (bx % by);
}

int32_t signedmodbl(int32_t x, int32_t y){

uint32_t isXNeg = (x <  (int32_t)0);

uint32_t isYNeg = (y <  (int32_t)0);

uint32_t ux = isXNeg ? ( (int32_t)0 - x) : x;

uint32_t uy = isYNeg ? ( (int32_t)0 - y) : y;

uint32_t urem = (ux % uy);
return isXNeg ? ( (uint32_t)0 - urem) : urem;
}

int32_t signedmodal(int32_t x, int32_t y){

int32_t bx = x;

int32_t by = y;
return (bx % by);
}


void MatMulCSF2DEigen(int32_t i, int32_t j, int32_t k, auto& A, auto& B, auto& C, int32_t consSF){
	Matrix<int64_t, Dynamic, Dynamic> eigen_a(i, j);
	Matrix<int64_t, Dynamic, Dynamic> eigen_b(j, k);
	Matrix<int64_t, Dynamic, Dynamic> eigen_c(i, k);

	for (int i0 = 0; i0 < i; ++i0){
		for (int i1 = 0; i1 < j; ++i1){
			eigen_a(i0, i1) = (int64_t)(A[i0][i1]);
		}
	}

	for (int i0 = 0; i0 < j; ++i0){
		for (int i1 = 0; i1 < k; ++i1){
			eigen_b(i0, i1) = (int64_t)(B[i0][i1]);
		}
	}

	eigen_c = eigen_a * eigen_b;

	for (int i0 = 0; i0 < i; ++i0){
		for (int i1 = 0; i1 < k; ++i1){
			C[i0][i1] = (int32_t)(eigen_c(i0, i1) >> consSF);
		}
	}
}

void MatMulCSF2D(int32_t i, int32_t j, int32_t k, auto& A, auto& B, auto& C, int32_t consSF){
// for (uint32_t i1 =  (int32_t)0; i1 < i; i1++){
// for (uint32_t i2 =  (int32_t)0; i2 < k; i2++){
// C[i1][i2] =  (int32_t)0;
// for (uint32_t i3 =  (int32_t)0; i3 < j; i3++){
// C[i1][i2] = (C[i1][i2] + (A[i1][i3] * B[i3][i2]));
// }
// C[i1][i2] = (C[i1][i2] >> consSF);
// }
// }
	MatMulCSF2DEigen(i,j,k,A,B,C,consSF);
}

void ArgMax1(int32_t outArrS1, int32_t inArrS1, int32_t inArrS2, auto& inArr, int32_t dim, auto& outArr){
for (uint32_t od =  (int32_t)0; od < inArrS1; od++){

int32_t maxi = inArr[od][ (int32_t)0];

int32_t maxiIdx =  (int32_t)0;
for (uint32_t i =  (int32_t)0; i < inArrS2; i++){

int32_t iL = i;
maxiIdx = (inArr[od][i] > maxi) ? iL : maxiIdx;
maxi = (inArr[od][i] > maxi) ? inArr[od][i] : maxi;
}
outArr[od] = maxiIdx;
}
}

void Relu2(int32_t s1, int32_t s2, auto& inArr, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = (inArr[i1][i2] >  (int32_t)0) ? inArr[i1][i2] :  (int32_t)0;
}
}
}

void Relu4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& inArr, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
outArr[i1][i2][i3][i4] = (inArr[i1][i2][i3][i4] >  (int32_t)0) ? inArr[i1][i2][i3][i4] :  (int32_t)0;
}
}
}
}
}

void ElemWiseMul2(int32_t s1, int32_t s2, auto& arr1, auto& arr2, auto& outArr, int32_t shrout){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = ((arr1[i1][i2] * arr2[i1][i2]) >> shrout);
}
}
}

void ElemWiseMul4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& arr1, auto& arr2, auto& outArr, int32_t shrout){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
outArr[i1][i2][i3][i4] = ((arr1[i1][i2][i3][i4] * arr2[i1][i2][i3][i4]) >> shrout);
}
}
}
}
}

void ElemWiseDiv2(int32_t s1, int32_t s2, auto& arr1, auto& arr2, auto& outArr, int32_t shrout){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = ((arr1[i1][i2] / arr2[i1][i2]) << shrout);
}
}
}

void Floor2(int32_t s1, int32_t s2, auto& inArr, auto& outArr, int32_t curSF){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){

int32_t mask = ~ (( (int32_t)1 << curSF) -  (int32_t)1);
outArr[i1][i2] = (inArr[i1][i2] & mask);
}
}
}

void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH, int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW, int32_t C1, auto& inArr, auto& outArr){
for (uint32_t n =  (int32_t)0; n < N; n++){
for (uint32_t c =  (int32_t)0; c < C; c++){

int32_t leftTopCornerH = ( (int32_t)0 - zPadHLeft);

int32_t extremeRightBottomCornerH = ((imgH -  (int32_t)1) + zPadHRight);

int32_t ctH =  (int32_t)0;
while ((((leftTopCornerH + ksizeH) -  (int32_t)1) <= extremeRightBottomCornerH)) {

int32_t leftTopCornerW = ( (int32_t)0 - zPadWLeft);

int32_t extremeRightBottomCornerW = ((imgW -  (int32_t)1) + zPadWRight);

int32_t ctW =  (int32_t)0;
while ((((leftTopCornerW + ksizeW) -  (int32_t)1) <= extremeRightBottomCornerW)) {

int32_t maxi =  (int32_t)0;
if ((((leftTopCornerH <  (int32_t)0) || (leftTopCornerH >= imgH)) || ((leftTopCornerW <  (int32_t)0) || (leftTopCornerW >= imgW)))) {
maxi =  (int32_t)0;
} else {
maxi = inArr[n][leftTopCornerH][leftTopCornerW][c];
}
for (uint32_t fh =  (int32_t)0; fh < ksizeH; fh++){
for (uint32_t fw =  (int32_t)0; fw < ksizeW; fw++){

int32_t curPosH = (leftTopCornerH + fh);

int32_t curPosW = (leftTopCornerW + fw);

int32_t temp =  (int32_t)0;
if ((((curPosH <  (int32_t)0) || (curPosH >= imgH)) || ((curPosW <  (int32_t)0) || (curPosW >= imgW)))) {
temp =  (int32_t)0;
} else {
temp = inArr[n][curPosH][curPosW][c];
}
maxi = (maxi < temp) ? temp : maxi;
}
}
outArr[n][ctH][ctW][c] = maxi;
leftTopCornerW = (leftTopCornerW + strideW);
ctW = (ctW +  (int32_t)1);
}

leftTopCornerH = (leftTopCornerH + strideH);
ctH = (ctH +  (int32_t)1);
}

}
}
}

void AvgPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH, int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW, int32_t C1, auto& inArr, auto& outArr){

int32_t rows = (((N * C) * H) * W);

auto filterAvg = make_vector<int32_t>(rows);

int32_t rowIdx =  (int32_t)0;
for (uint32_t n =  (int32_t)0; n < N; n++){
for (uint32_t c =  (int32_t)0; c < C; c++){

int32_t leftTopCornerH = ( (int32_t)0 - zPadHLeft);

int32_t extremeRightBottomCornerH = ((imgH -  (int32_t)1) + zPadHRight);

int32_t ctH =  (int32_t)0;
while ((((leftTopCornerH + ksizeH) -  (int32_t)1) <= extremeRightBottomCornerH)) {

int32_t leftTopCornerW = ( (int32_t)0 - zPadWLeft);

int32_t extremeRightBottomCornerW = ((imgW -  (int32_t)1) + zPadWRight);

int32_t ctW =  (int32_t)0;
while ((((leftTopCornerW + ksizeW) -  (int32_t)1) <= extremeRightBottomCornerW)) {

int32_t curFilterSum =  (int32_t)0;
for (uint32_t fh =  (int32_t)0; fh < ksizeH; fh++){
for (uint32_t fw =  (int32_t)0; fw < ksizeW; fw++){

int32_t curPosH = (leftTopCornerH + fh);

int32_t curPosW = (leftTopCornerW + fw);

int32_t temp =  (int32_t)0;
if ((((curPosH <  (int32_t)0) || (curPosH >= imgH)) || ((curPosW <  (int32_t)0) || (curPosW >= imgW)))) {
temp =  (int32_t)0;
} else {
temp = inArr[n][curPosH][curPosW][c];
}
curFilterSum = (curFilterSum + temp);
}
}

int32_t ksizeH64 = ksizeH;

int32_t ksizeW64 = ksizeW;

int32_t filterSz64 = (ksizeH64 * ksizeW64);

int32_t curFilterAvg = (curFilterSum / filterSz64);
filterAvg[rowIdx] = curFilterAvg;
rowIdx = (rowIdx +  (int32_t)1);
leftTopCornerW = (leftTopCornerW + strideW);
ctW = (ctW +  (int32_t)1);
}

leftTopCornerH = (leftTopCornerH + strideH);
ctH = (ctH +  (int32_t)1);
}

}
}
for (uint32_t n =  (int32_t)0; n < N; n++){
for (uint32_t c =  (int32_t)0; c < C; c++){
for (uint32_t h =  (int32_t)0; h < H; h++){
for (uint32_t w =  (int32_t)0; w < W; w++){
outArr[n][h][w][c] = filterAvg[((((((n * C) * H) * W) + ((c * H) * W)) + (h * W)) + w)];
}
}
}
}
}

void TempFusedBatchNorm4411(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& inArr, int32_t vecS1, auto& multArr, auto& biasArr, auto& outputArr, int32_t consSF){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){

int32_t t1 = (inArr[i1][i2][i3][i4] * multArr[i4]);

int32_t t2 = (t1 >>  consSF);
outputArr[i1][i2][i3][i4] = (t2 + biasArr[i4]);
}
}
}
}
}

void ScalarMul4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t scalar, auto& inputArr, auto& outputArr, int32_t consSF){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
outputArr[i1][i2][i3][i4] = (inputArr[i1][i2][i3][i4] * scalar);
}
}
}
}
}

void ReduceMean24(int32_t outS1, int32_t outS2, int32_t inS1, int32_t inS2, int32_t inS3, int32_t inS4, auto& inputArr, auto& axes, auto& outputArr){
for (uint32_t i1 =  (int32_t)0; i1 < outS1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < outS2; i2++){

int32_t summ =  (int32_t)0;
for (uint32_t i =  (int32_t)0; i < inS2; i++){
for (uint32_t j =  (int32_t)0; j < inS3; j++){
summ = (summ + inputArr[i1][i][j][i2]);
}
}

int32_t numElem = (inS2 * inS3);
summ = (summ / numElem);
outputArr[i1][i2] = summ;
}
}
}

void MatAddBroadCast2(int32_t s1, int32_t s2, auto& A, auto& B, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = (A[i1][i2] + B[i2]);
}
}
}

void MatAdd2(int32_t s1, int32_t s2, auto& A, auto& B, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = (A[i1][i2] + B[i1][i2]);
}
}
}

void MatAddBroadCast4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& A, auto& B, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
outArr[i1][i2][i3][i4] = (A[i1][i2][i3][i4] + B[i4]);
}
}
}
}
}

void MatAdd4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& A, auto& B, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
outArr[i1][i2][i3][i4] = (A[i1][i2][i3][i4] + B[i1][i2][i3][i4]);
}
}
}
}
}

void CreateTensor1(int32_t s1, int32_t val, auto& arr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
arr[i1] = val;
}
}

void CreateTensor2(int32_t s1, int32_t s2, int32_t val, auto& arr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
arr[i1][i2] = val;
}
}
}

void CreateTensor4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t val, auto& arr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
arr[i1][i2][i3][i4] = val;
}
}
}
}
}

void CopyTensor1(int32_t s1, auto& targetArr, auto& fromArr, auto& ignore){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
targetArr[i1] = fromArr[i1];
}
}

void CopyTensor2(int32_t s1, int32_t s2, auto& targetArr, auto& fromArr, auto& ignore){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
targetArr[i1][i2] = fromArr[i1][i2];
}
}
}

void CopyTensor4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& targetArr, auto& fromArr, auto& ignore){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
targetArr[i1][i2][i3][i4] = fromArr[i1][i2][i3][i4];
}
}
}
}
}

void CreateIdentity11(int32_t s1, auto& fromArr, auto& newArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
newArr[i1] = fromArr[i1];
}
}

void CreateIdentity22(int32_t s1, int32_t s2, auto& fromArr, auto& newArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
newArr[i1][i2] = fromArr[i1][i2];
}
}
}

void CreateIdentity44(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& fromArr, auto& newArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
newArr[i1][i2][i3][i4] = fromArr[i1][i2][i3][i4];
}
}
}
}
}

void Concat2T444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto& inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto& inp2, int32_t axis, auto& outp){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
if ((axis ==  (int32_t)0)) {
if ((i1 < inp1s1)) {
outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
} else {
outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
}
} else {
if ((axis ==  (int32_t)1)) {
if ((i2 < inp1s2)) {
outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
} else {
outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
}
} else {
if ((axis ==  (int32_t)2)) {
if ((i3 < inp1s3)) {
outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
} else {
outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
}
} else {
if ((i4 < inp1s4)) {
outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
} else {
outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
}
}
}
}
}
}
}
}
}

void RandomUniform2(int32_t s1, int32_t s2, int32_t dataType, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] =  (int32_t)100;
}
}
}

void Conv2DReshapeFilter(int32_t FH, int32_t FW, int32_t CI, int32_t CO, auto& inputArr, auto& outputArr){
for (uint32_t co =  (int32_t)0; co < CO; co++){
for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
for (uint32_t fw =  (int32_t)0; fw < FW; fw++){
for (uint32_t ci =  (int32_t)0; ci < CI; ci++){

int32_t linIdx = ((((fh * FW) * CI) + (fw * CI)) + ci);
outputArr[co][linIdx] = inputArr[fh][fw][ci][co];
}
}
}
}
}

void Conv2DReshapeMatMulOP(int32_t N, int32_t finalH, int32_t finalW, int32_t CO, auto& inputArr, auto& outputArr){
for (uint32_t co =  (int32_t)0; co < CO; co++){
for (uint32_t n =  (int32_t)0; n < N; n++){
for (uint32_t h =  (int32_t)0; h < finalH; h++){
for (uint32_t w =  (int32_t)0; w < finalW; w++){
outputArr[n][h][w][co] = inputArr[co][((((n * finalH) * finalW) + (h * finalW)) + w)];
}
}
}
}
}

void Conv2DReshapeInput(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t RRows, int32_t RCols, auto& inputArr, auto& outputArr){

int32_t linIdxFilterMult =  (int32_t)0;
for (uint32_t n =  (int32_t)0; n < N; n++){

int32_t leftTopCornerH = ( (int32_t)0 - zPadHLeft);

int32_t extremeRightBottomCornerH = ((H -  (int32_t)1) + zPadHRight);
while ((((leftTopCornerH + FH) -  (int32_t)1) <= extremeRightBottomCornerH)) {

int32_t leftTopCornerW = ( (int32_t)0 - zPadWLeft);

int32_t extremeRightBottomCornerW = ((W -  (int32_t)1) + zPadWRight);
while ((((leftTopCornerW + FW) -  (int32_t)1) <= extremeRightBottomCornerW)) {
for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
for (uint32_t fw =  (int32_t)0; fw < FW; fw++){

int32_t curPosH = (leftTopCornerH + fh);

int32_t curPosW = (leftTopCornerW + fw);

int32_t val =  (int32_t)0;
for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
if ((((curPosH <  (int32_t)0) || (curPosH >= H)) || ((curPosW <  (int32_t)0) || (curPosW >= W)))) {
val =  (int32_t)0;
} else {
val = inputArr[n][curPosH][curPosW][ci];
}
outputArr[((((fh * FW) * CI) + (fw * CI)) + ci)][linIdxFilterMult] = val;
}
}
}
linIdxFilterMult = (linIdxFilterMult +  (int32_t)1);
leftTopCornerW = (leftTopCornerW + strideW);
}

leftTopCornerH = (leftTopCornerH + strideH);
}

}
}

void Conv2DCSF(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, auto& inputArr, auto& filterArr, auto& outArr, int32_t consSF){

int32_t reshapedFilterRows = CO;

int32_t reshapedFilterCols = ((FH * FW) * CI);

int32_t reshapedIPRows = ((FH * FW) * CI);

int32_t newH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) +  (int32_t)1);

int32_t newW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) +  (int32_t)1);

int32_t reshapedIPCols = ((N * newH) * newW);

auto filterReshaped = make_vector<int32_t>(reshapedFilterRows, reshapedFilterCols);

auto inputReshaped = make_vector<int32_t>(reshapedIPRows, reshapedIPCols);

auto matmulOP = make_vector<int32_t>(reshapedFilterRows, reshapedIPCols);
Conv2DReshapeFilter(FH, FW, CI, CO, filterArr, filterReshaped);
Conv2DReshapeInput(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
MatMulCSF2D(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP, consSF);
Conv2DReshapeMatMulOP(N, newH, newW, CO, matmulOP, outArr);
}

void Transpose2(int32_t s1, int32_t s2, auto& inArr, auto& outArr){
for (uint32_t i =  (int32_t)0; i < s1; i++){
for (uint32_t j =  (int32_t)0; j < s2; j++){
outArr[i][j] = inArr[j][i];
}
}
}

void Pad442(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inps1, int32_t inps2, int32_t inps3, int32_t inps4, auto& inpArr, int32_t pads1, int32_t pads2, auto& paddings, auto& outArr){

int32_t lbounds1 = paddings[ (int32_t)0][ (int32_t)0];

int32_t rbounds1excl = (s1 - paddings[ (int32_t)0][ (int32_t)1]);

int32_t lbounds2 = paddings[ (int32_t)1][ (int32_t)0];

int32_t rbounds2excl = (s2 - paddings[ (int32_t)1][ (int32_t)1]);

int32_t lbounds3 = paddings[ (int32_t)2][ (int32_t)0];

int32_t rbounds3excl = (s3 - paddings[ (int32_t)2][ (int32_t)1]);

int32_t lbounds4 = paddings[ (int32_t)3][ (int32_t)0];

int32_t rbounds4excl = (s4 - paddings[ (int32_t)3][ (int32_t)1]);
for (uint32_t i =  (int32_t)0; i < s1; i++){
for (uint32_t j =  (int32_t)0; j < s2; j++){
for (uint32_t k =  (int32_t)0; k < s3; k++){
for (uint32_t l =  (int32_t)0; l < s4; l++){
if (((((((((i >= lbounds1) && (i < rbounds1excl)) && (j >= lbounds2)) && (j < rbounds2excl)) && (k >= lbounds3)) && (k < rbounds3excl)) && (l >= lbounds4)) && (l < rbounds4excl))) {
outArr[i][j][k][l] = inpArr[(i - paddings[ (int32_t)0][ (int32_t)0])][(j - paddings[ (int32_t)1][ (int32_t)0])][(k - paddings[ (int32_t)2][ (int32_t)0])][(l - paddings[ (int32_t)3][ (int32_t)0])];
} else {
outArr[i][j][k][l] =  (int32_t)0;
}
}
}
}
}
}

void Squeeze24(int32_t s1, int32_t s2, int32_t dim1, int32_t dim2, int32_t ins1, int32_t ins2, int32_t ins3, int32_t ins4, auto& inArr, auto& outArr){
for (uint32_t i =  (int32_t)0; i < ins1; i++){
for (uint32_t j =  (int32_t)0; j < ins2; j++){
for (uint32_t k =  (int32_t)0; k < ins3; k++){
for (uint32_t l =  (int32_t)0; l < ins4; l++){

int32_t linIdx = ((((((i * ins2) * ins3) * ins4) + ((j * ins3) * ins4)) + (k * ins4)) + l);

int32_t outIdx1 = (linIdx / s2);

int32_t outIdx2 = (linIdx % s2);
outArr[outIdx1][outIdx2] = inArr[i][j][k][l];
}
}
}
}
}

void readIdxFromRandomSubsetFile(string idxFile, int M, int acutalImgIdx[]){
	ifstream filep(idxFile);
	string str;
	int ct = 0;
	while(getline(filep,str)){
		 acutalImgIdx[ct++] = atoi(str.c_str());
		 if (ct >= M){
		 	break;
		 }
	}
	if (ct!=M){
		assert(false);
	}
}

int main (int argc, char** argv) {
ios_base::sync_with_stdio(false);
if ((argc != 5) && (argc != 7)){
	cerr<<"Incorrect args provided."<<endl;
	exit(1);
}
int consSF = atoi(argv[1]);
int startImgNum = atoi(argv[2]);
int endImgNum = atoi(argv[3]);
string preProcessedImgDir = string(argv[4]);
int randomSubsetNumImages = 1;
string randomSubsetIdxTestFileName = "";
if (argc == 7){
	randomSubsetNumImages = atoi(argv[5]);
	randomSubsetIdxTestFileName = string(argv[6]);
}

if ((preProcessedImgDir[preProcessedImgDir.length()-1] == '/') 
	|| (randomSubsetIdxTestFileName[randomSubsetIdxTestFileName.length()-1] == '/')){
	cerr<<"Paths provided shouldn't have / at their end."<<endl;
	exit(1);
}

if (startImgNum==0){
	cerr<<"Start img number should be 1-indexed."<<endl;
	exit(1);
}

auto tmp252 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp253 = make_vector<int32_t>( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3);

auto tmp254 = make_vector<int32_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64);

auto tmp255 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp256 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp257 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp258 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp259 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp260 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp261 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp262 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp263 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp264 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp265 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp266 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp267 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp268 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp269 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp270 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp271 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp272 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp273 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp274 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp275 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp276 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp277 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp278 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp279 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp280 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp281 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp282 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp283 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp284 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp285 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp286 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp287 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp288 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp289 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp290 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp291 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp292 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp293 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp294 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp295 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp296 = make_vector<int32_t>( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128);

auto tmp297 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp298 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp299 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp300 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp301 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp302 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp303 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp304 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp305 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp306 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp307 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp308 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp309 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp310 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp311 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp312 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp313 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp314 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp315 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp316 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp317 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp318 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp319 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp320 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp321 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp322 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp323 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp324 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp325 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp326 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp327 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp328 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp329 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp330 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp331 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp332 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp333 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp334 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp335 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp336 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp337 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);

auto tmp338 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);

auto tmp339 = make_vector<int32_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);

auto tmp340 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp341 = make_vector<int32_t>( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256);

auto tmp342 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp343 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp344 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp345 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp346 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp347 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp348 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp349 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp350 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp351 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp352 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp353 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp354 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp355 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp356 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp357 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp358 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp359 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp360 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp361 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp362 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp363 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp364 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp365 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp366 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp367 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp368 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp369 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp370 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp371 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp372 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp373 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp374 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp375 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp376 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp377 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp378 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp379 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp380 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp381 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp382 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp383 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp384 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp385 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp386 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp387 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp388 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp389 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp390 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp391 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp392 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp393 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp394 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp395 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp396 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp397 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp398 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp399 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp400 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp401 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp402 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);

auto tmp403 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);

auto tmp404 = make_vector<int32_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);

auto tmp405 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp406 = make_vector<int32_t>( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512);

auto tmp407 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp408 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp409 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp410 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp411 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp412 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp413 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp414 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp415 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp416 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp417 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp418 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp419 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp420 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp421 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp422 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp423 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp424 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp425 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp426 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp427 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp428 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp429 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp430 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp431 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp432 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp433 = make_vector<int32_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp434 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048);

auto tmp435 = make_vector<int32_t>( (int32_t)1,  (int32_t)2048);

auto tmp436 = make_vector<int32_t>( (int32_t)1,  (int32_t)1001);

auto tmp437 = make_vector<int32_t>( (int32_t)1,  (int32_t)1001);

// No need of argmax - so that topKacc can be calculated
// auto tmp438 = make_vector<int32_t>( (int32_t)1);




auto tmp1 = make_vector<int32_t>( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp1 at (865,1-865,44) */
long double __tmp_in_tmp1;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)7; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)7; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)3; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp1;
tmp1[i0][i1][i2][i3] = ldexp(__tmp_in_tmp1, consSF);
}
}
}
}

auto tmp2 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp2 at (867,1-867,35) */
long double __tmp_in_tmp2;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp2;
tmp2[i0] = ldexp(__tmp_in_tmp2, consSF);
}

auto tmp3 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp3 at (869,1-869,35) */
long double __tmp_in_tmp3;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp3;
tmp3[i0] = ldexp(__tmp_in_tmp3, consSF);
}

auto tmp4 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp4 at (871,1-871,35) */
long double __tmp_in_tmp4;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp4;
tmp4[i0] = ldexp(__tmp_in_tmp4, consSF);
}

auto tmp5 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp5 at (873,1-873,35) */
long double __tmp_in_tmp5;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp5;
tmp5[i0] = ldexp(__tmp_in_tmp5, consSF);
}

auto tmp6 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp6 at (875,1-875,46) */
long double __tmp_in_tmp6;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp6;
tmp6[i0][i1][i2][i3] = ldexp(__tmp_in_tmp6, consSF);
}
}
}
}

auto tmp7 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp7 at (877,1-877,45) */
long double __tmp_in_tmp7;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp7;
tmp7[i0][i1][i2][i3] = ldexp(__tmp_in_tmp7, consSF);
}
}
}
}

auto tmp8 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp8 at (879,1-879,35) */
long double __tmp_in_tmp8;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp8;
tmp8[i0] = ldexp(__tmp_in_tmp8, consSF);
}

auto tmp9 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp9 at (881,1-881,35) */
long double __tmp_in_tmp9;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp9;
tmp9[i0] = ldexp(__tmp_in_tmp9, consSF);
}

auto tmp10 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp10 at (883,1-883,36) */
long double __tmp_in_tmp10;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp10;
tmp10[i0] = ldexp(__tmp_in_tmp10, consSF);
}

auto tmp11 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp11 at (885,1-885,36) */
long double __tmp_in_tmp11;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp11;
tmp11[i0] = ldexp(__tmp_in_tmp11, consSF);
}

auto tmp12 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp12 at (887,1-887,46) */
long double __tmp_in_tmp12;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp12;
tmp12[i0][i1][i2][i3] = ldexp(__tmp_in_tmp12, consSF);
}
}
}
}

auto tmp13 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp13 at (889,1-889,36) */
long double __tmp_in_tmp13;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp13;
tmp13[i0] = ldexp(__tmp_in_tmp13, consSF);
}

auto tmp14 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp14 at (891,1-891,36) */
long double __tmp_in_tmp14;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp14;
tmp14[i0] = ldexp(__tmp_in_tmp14, consSF);
}

auto tmp15 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp15 at (893,1-893,36) */
long double __tmp_in_tmp15;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp15;
tmp15[i0] = ldexp(__tmp_in_tmp15, consSF);
}

auto tmp16 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp16 at (895,1-895,36) */
long double __tmp_in_tmp16;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp16;
tmp16[i0] = ldexp(__tmp_in_tmp16, consSF);
}

auto tmp17 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp17 at (897,1-897,47) */
long double __tmp_in_tmp17;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp17;
tmp17[i0][i1][i2][i3] = ldexp(__tmp_in_tmp17, consSF);
}
}
}
}

auto tmp18 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp18 at (899,1-899,37) */
long double __tmp_in_tmp18;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp18;
tmp18[i0] = ldexp(__tmp_in_tmp18, consSF);
}

auto tmp19 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp19 at (901,1-901,37) */
long double __tmp_in_tmp19;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp19;
tmp19[i0] = ldexp(__tmp_in_tmp19, consSF);
}

auto tmp20 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp20 at (903,1-903,37) */
long double __tmp_in_tmp20;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp20;
tmp20[i0] = ldexp(__tmp_in_tmp20, consSF);
}

auto tmp21 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp21 at (905,1-905,37) */
long double __tmp_in_tmp21;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp21;
tmp21[i0] = ldexp(__tmp_in_tmp21, consSF);
}

auto tmp22 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp22 at (907,1-907,47) */
long double __tmp_in_tmp22;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp22;
tmp22[i0][i1][i2][i3] = ldexp(__tmp_in_tmp22, consSF);
}
}
}
}

auto tmp23 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp23 at (909,1-909,36) */
long double __tmp_in_tmp23;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp23;
tmp23[i0] = ldexp(__tmp_in_tmp23, consSF);
}

auto tmp24 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp24 at (911,1-911,36) */
long double __tmp_in_tmp24;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp24;
tmp24[i0] = ldexp(__tmp_in_tmp24, consSF);
}

auto tmp25 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp25 at (913,1-913,36) */
long double __tmp_in_tmp25;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp25;
tmp25[i0] = ldexp(__tmp_in_tmp25, consSF);
}

auto tmp26 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp26 at (915,1-915,36) */
long double __tmp_in_tmp26;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp26;
tmp26[i0] = ldexp(__tmp_in_tmp26, consSF);
}

auto tmp27 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp27 at (917,1-917,46) */
long double __tmp_in_tmp27;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp27;
tmp27[i0][i1][i2][i3] = ldexp(__tmp_in_tmp27, consSF);
}
}
}
}

auto tmp28 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp28 at (919,1-919,36) */
long double __tmp_in_tmp28;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp28;
tmp28[i0] = ldexp(__tmp_in_tmp28, consSF);
}

auto tmp29 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp29 at (921,1-921,36) */
long double __tmp_in_tmp29;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp29;
tmp29[i0] = ldexp(__tmp_in_tmp29, consSF);
}

auto tmp30 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp30 at (923,1-923,36) */
long double __tmp_in_tmp30;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp30;
tmp30[i0] = ldexp(__tmp_in_tmp30, consSF);
}

auto tmp31 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp31 at (925,1-925,36) */
long double __tmp_in_tmp31;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp31;
tmp31[i0] = ldexp(__tmp_in_tmp31, consSF);
}

auto tmp32 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp32 at (927,1-927,47) */
long double __tmp_in_tmp32;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp32;
tmp32[i0][i1][i2][i3] = ldexp(__tmp_in_tmp32, consSF);
}
}
}
}

auto tmp33 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp33 at (929,1-929,37) */
long double __tmp_in_tmp33;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp33;
tmp33[i0] = ldexp(__tmp_in_tmp33, consSF);
}

auto tmp34 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp34 at (931,1-931,37) */
long double __tmp_in_tmp34;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp34;
tmp34[i0] = ldexp(__tmp_in_tmp34, consSF);
}

auto tmp35 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp35 at (933,1-933,37) */
long double __tmp_in_tmp35;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp35;
tmp35[i0] = ldexp(__tmp_in_tmp35, consSF);
}

auto tmp36 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp36 at (935,1-935,37) */
long double __tmp_in_tmp36;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp36;
tmp36[i0] = ldexp(__tmp_in_tmp36, consSF);
}

auto tmp37 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp37 at (937,1-937,47) */
long double __tmp_in_tmp37;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp37;
tmp37[i0][i1][i2][i3] = ldexp(__tmp_in_tmp37, consSF);
}
}
}
}

auto tmp38 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp38 at (939,1-939,36) */
long double __tmp_in_tmp38;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp38;
tmp38[i0] = ldexp(__tmp_in_tmp38, consSF);
}

auto tmp39 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp39 at (941,1-941,36) */
long double __tmp_in_tmp39;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp39;
tmp39[i0] = ldexp(__tmp_in_tmp39, consSF);
}

auto tmp40 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp40 at (943,1-943,36) */
long double __tmp_in_tmp40;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp40;
tmp40[i0] = ldexp(__tmp_in_tmp40, consSF);
}

auto tmp41 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp41 at (945,1-945,36) */
long double __tmp_in_tmp41;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp41;
tmp41[i0] = ldexp(__tmp_in_tmp41, consSF);
}

auto tmp42 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp42 at (947,1-947,46) */
long double __tmp_in_tmp42;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp42;
tmp42[i0][i1][i2][i3] = ldexp(__tmp_in_tmp42, consSF);
}
}
}
}

auto tmp43 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp43 at (949,1-949,36) */
long double __tmp_in_tmp43;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp43;
tmp43[i0] = ldexp(__tmp_in_tmp43, consSF);
}

auto tmp44 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp44 at (951,1-951,36) */
long double __tmp_in_tmp44;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp44;
tmp44[i0] = ldexp(__tmp_in_tmp44, consSF);
}

auto tmp45 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp45 at (953,1-953,36) */
long double __tmp_in_tmp45;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp45;
tmp45[i0] = ldexp(__tmp_in_tmp45, consSF);
}

auto tmp46 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp46 at (955,1-955,36) */
long double __tmp_in_tmp46;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp46;
tmp46[i0] = ldexp(__tmp_in_tmp46, consSF);
}

auto tmp47 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp47 at (957,1-957,47) */
long double __tmp_in_tmp47;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp47;
tmp47[i0][i1][i2][i3] = ldexp(__tmp_in_tmp47, consSF);
}
}
}
}

auto tmp48 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp48 at (959,1-959,37) */
long double __tmp_in_tmp48;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp48;
tmp48[i0] = ldexp(__tmp_in_tmp48, consSF);
}

auto tmp49 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp49 at (961,1-961,37) */
long double __tmp_in_tmp49;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp49;
tmp49[i0] = ldexp(__tmp_in_tmp49, consSF);
}

auto tmp50 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp50 at (963,1-963,37) */
long double __tmp_in_tmp50;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp50;
tmp50[i0] = ldexp(__tmp_in_tmp50, consSF);
}

auto tmp51 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp51 at (965,1-965,37) */
long double __tmp_in_tmp51;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp51;
tmp51[i0] = ldexp(__tmp_in_tmp51, consSF);
}

auto tmp52 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp52 at (967,1-967,48) */
long double __tmp_in_tmp52;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp52;
tmp52[i0][i1][i2][i3] = ldexp(__tmp_in_tmp52, consSF);
}
}
}
}

auto tmp53 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp53 at (969,1-969,48) */
long double __tmp_in_tmp53;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp53;
tmp53[i0][i1][i2][i3] = ldexp(__tmp_in_tmp53, consSF);
}
}
}
}

auto tmp54 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp54 at (971,1-971,37) */
long double __tmp_in_tmp54;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp54;
tmp54[i0] = ldexp(__tmp_in_tmp54, consSF);
}

auto tmp55 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp55 at (973,1-973,37) */
long double __tmp_in_tmp55;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp55;
tmp55[i0] = ldexp(__tmp_in_tmp55, consSF);
}

auto tmp56 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp56 at (975,1-975,37) */
long double __tmp_in_tmp56;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp56;
tmp56[i0] = ldexp(__tmp_in_tmp56, consSF);
}

auto tmp57 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp57 at (977,1-977,37) */
long double __tmp_in_tmp57;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp57;
tmp57[i0] = ldexp(__tmp_in_tmp57, consSF);
}

auto tmp58 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp58 at (979,1-979,48) */
long double __tmp_in_tmp58;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp58;
tmp58[i0][i1][i2][i3] = ldexp(__tmp_in_tmp58, consSF);
}
}
}
}

auto tmp59 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp59 at (981,1-981,37) */
long double __tmp_in_tmp59;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp59;
tmp59[i0] = ldexp(__tmp_in_tmp59, consSF);
}

auto tmp60 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp60 at (983,1-983,37) */
long double __tmp_in_tmp60;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp60;
tmp60[i0] = ldexp(__tmp_in_tmp60, consSF);
}

auto tmp61 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp61 at (985,1-985,37) */
long double __tmp_in_tmp61;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp61;
tmp61[i0] = ldexp(__tmp_in_tmp61, consSF);
}

auto tmp62 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp62 at (987,1-987,37) */
long double __tmp_in_tmp62;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp62;
tmp62[i0] = ldexp(__tmp_in_tmp62, consSF);
}

auto tmp63 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp63 at (989,1-989,48) */
long double __tmp_in_tmp63;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp63;
tmp63[i0][i1][i2][i3] = ldexp(__tmp_in_tmp63, consSF);
}
}
}
}

auto tmp64 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp64 at (991,1-991,37) */
long double __tmp_in_tmp64;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp64;
tmp64[i0] = ldexp(__tmp_in_tmp64, consSF);
}

auto tmp65 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp65 at (993,1-993,37) */
long double __tmp_in_tmp65;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp65;
tmp65[i0] = ldexp(__tmp_in_tmp65, consSF);
}

auto tmp66 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp66 at (995,1-995,37) */
long double __tmp_in_tmp66;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp66;
tmp66[i0] = ldexp(__tmp_in_tmp66, consSF);
}

auto tmp67 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp67 at (997,1-997,37) */
long double __tmp_in_tmp67;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp67;
tmp67[i0] = ldexp(__tmp_in_tmp67, consSF);
}

auto tmp68 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp68 at (999,1-999,48) */
long double __tmp_in_tmp68;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp68;
tmp68[i0][i1][i2][i3] = ldexp(__tmp_in_tmp68, consSF);
}
}
}
}

auto tmp69 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp69 at (1001,1-1001,37) */
long double __tmp_in_tmp69;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp69;
tmp69[i0] = ldexp(__tmp_in_tmp69, consSF);
}

auto tmp70 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp70 at (1003,1-1003,37) */
long double __tmp_in_tmp70;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp70;
tmp70[i0] = ldexp(__tmp_in_tmp70, consSF);
}

auto tmp71 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp71 at (1005,1-1005,37) */
long double __tmp_in_tmp71;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp71;
tmp71[i0] = ldexp(__tmp_in_tmp71, consSF);
}

auto tmp72 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp72 at (1007,1-1007,37) */
long double __tmp_in_tmp72;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp72;
tmp72[i0] = ldexp(__tmp_in_tmp72, consSF);
}

auto tmp73 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp73 at (1009,1-1009,48) */
long double __tmp_in_tmp73;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp73;
tmp73[i0][i1][i2][i3] = ldexp(__tmp_in_tmp73, consSF);
}
}
}
}

auto tmp74 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp74 at (1011,1-1011,37) */
long double __tmp_in_tmp74;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp74;
tmp74[i0] = ldexp(__tmp_in_tmp74, consSF);
}

auto tmp75 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp75 at (1013,1-1013,37) */
long double __tmp_in_tmp75;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp75;
tmp75[i0] = ldexp(__tmp_in_tmp75, consSF);
}

auto tmp76 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp76 at (1015,1-1015,37) */
long double __tmp_in_tmp76;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp76;
tmp76[i0] = ldexp(__tmp_in_tmp76, consSF);
}

auto tmp77 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp77 at (1017,1-1017,37) */
long double __tmp_in_tmp77;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp77;
tmp77[i0] = ldexp(__tmp_in_tmp77, consSF);
}

auto tmp78 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp78 at (1019,1-1019,48) */
long double __tmp_in_tmp78;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp78;
tmp78[i0][i1][i2][i3] = ldexp(__tmp_in_tmp78, consSF);
}
}
}
}

auto tmp79 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp79 at (1021,1-1021,37) */
long double __tmp_in_tmp79;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp79;
tmp79[i0] = ldexp(__tmp_in_tmp79, consSF);
}

auto tmp80 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp80 at (1023,1-1023,37) */
long double __tmp_in_tmp80;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp80;
tmp80[i0] = ldexp(__tmp_in_tmp80, consSF);
}

auto tmp81 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp81 at (1025,1-1025,37) */
long double __tmp_in_tmp81;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp81;
tmp81[i0] = ldexp(__tmp_in_tmp81, consSF);
}

auto tmp82 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp82 at (1027,1-1027,37) */
long double __tmp_in_tmp82;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp82;
tmp82[i0] = ldexp(__tmp_in_tmp82, consSF);
}

auto tmp83 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp83 at (1029,1-1029,48) */
long double __tmp_in_tmp83;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp83;
tmp83[i0][i1][i2][i3] = ldexp(__tmp_in_tmp83, consSF);
}
}
}
}

auto tmp84 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp84 at (1031,1-1031,37) */
long double __tmp_in_tmp84;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp84;
tmp84[i0] = ldexp(__tmp_in_tmp84, consSF);
}

auto tmp85 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp85 at (1033,1-1033,37) */
long double __tmp_in_tmp85;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp85;
tmp85[i0] = ldexp(__tmp_in_tmp85, consSF);
}

auto tmp86 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp86 at (1035,1-1035,37) */
long double __tmp_in_tmp86;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp86;
tmp86[i0] = ldexp(__tmp_in_tmp86, consSF);
}

auto tmp87 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp87 at (1037,1-1037,37) */
long double __tmp_in_tmp87;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp87;
tmp87[i0] = ldexp(__tmp_in_tmp87, consSF);
}

auto tmp88 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp88 at (1039,1-1039,48) */
long double __tmp_in_tmp88;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp88;
tmp88[i0][i1][i2][i3] = ldexp(__tmp_in_tmp88, consSF);
}
}
}
}

auto tmp89 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp89 at (1041,1-1041,37) */
long double __tmp_in_tmp89;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp89;
tmp89[i0] = ldexp(__tmp_in_tmp89, consSF);
}

auto tmp90 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp90 at (1043,1-1043,37) */
long double __tmp_in_tmp90;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp90;
tmp90[i0] = ldexp(__tmp_in_tmp90, consSF);
}

auto tmp91 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp91 at (1045,1-1045,37) */
long double __tmp_in_tmp91;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp91;
tmp91[i0] = ldexp(__tmp_in_tmp91, consSF);
}

auto tmp92 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp92 at (1047,1-1047,37) */
long double __tmp_in_tmp92;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp92;
tmp92[i0] = ldexp(__tmp_in_tmp92, consSF);
}

auto tmp93 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp93 at (1049,1-1049,48) */
long double __tmp_in_tmp93;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp93;
tmp93[i0][i1][i2][i3] = ldexp(__tmp_in_tmp93, consSF);
}
}
}
}

auto tmp94 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp94 at (1051,1-1051,37) */
long double __tmp_in_tmp94;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp94;
tmp94[i0] = ldexp(__tmp_in_tmp94, consSF);
}

auto tmp95 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp95 at (1053,1-1053,37) */
long double __tmp_in_tmp95;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp95;
tmp95[i0] = ldexp(__tmp_in_tmp95, consSF);
}

auto tmp96 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp96 at (1055,1-1055,37) */
long double __tmp_in_tmp96;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp96;
tmp96[i0] = ldexp(__tmp_in_tmp96, consSF);
}

auto tmp97 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp97 at (1057,1-1057,37) */
long double __tmp_in_tmp97;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp97;
tmp97[i0] = ldexp(__tmp_in_tmp97, consSF);
}

auto tmp98 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp98 at (1059,1-1059,48) */
long double __tmp_in_tmp98;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp98;
tmp98[i0][i1][i2][i3] = ldexp(__tmp_in_tmp98, consSF);
}
}
}
}

auto tmp99 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp99 at (1061,1-1061,37) */
long double __tmp_in_tmp99;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp99;
tmp99[i0] = ldexp(__tmp_in_tmp99, consSF);
}

auto tmp100 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp100 at (1063,1-1063,38) */
long double __tmp_in_tmp100;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp100;
tmp100[i0] = ldexp(__tmp_in_tmp100, consSF);
}

auto tmp101 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp101 at (1065,1-1065,38) */
long double __tmp_in_tmp101;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp101;
tmp101[i0] = ldexp(__tmp_in_tmp101, consSF);
}

auto tmp102 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp102 at (1067,1-1067,38) */
long double __tmp_in_tmp102;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp102;
tmp102[i0] = ldexp(__tmp_in_tmp102, consSF);
}

auto tmp103 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp103 at (1069,1-1069,49) */
long double __tmp_in_tmp103;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp103;
tmp103[i0][i1][i2][i3] = ldexp(__tmp_in_tmp103, consSF);
}
}
}
}

auto tmp104 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp104 at (1071,1-1071,38) */
long double __tmp_in_tmp104;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp104;
tmp104[i0] = ldexp(__tmp_in_tmp104, consSF);
}

auto tmp105 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp105 at (1073,1-1073,38) */
long double __tmp_in_tmp105;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp105;
tmp105[i0] = ldexp(__tmp_in_tmp105, consSF);
}

auto tmp106 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp106 at (1075,1-1075,38) */
long double __tmp_in_tmp106;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp106;
tmp106[i0] = ldexp(__tmp_in_tmp106, consSF);
}

auto tmp107 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp107 at (1077,1-1077,38) */
long double __tmp_in_tmp107;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp107;
tmp107[i0] = ldexp(__tmp_in_tmp107, consSF);
}

auto tmp108 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp108 at (1079,1-1079,49) */
long double __tmp_in_tmp108;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp108;
tmp108[i0][i1][i2][i3] = ldexp(__tmp_in_tmp108, consSF);
}
}
}
}

auto tmp109 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp109 at (1081,1-1081,38) */
long double __tmp_in_tmp109;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp109;
tmp109[i0] = ldexp(__tmp_in_tmp109, consSF);
}

auto tmp110 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp110 at (1083,1-1083,38) */
long double __tmp_in_tmp110;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp110;
tmp110[i0] = ldexp(__tmp_in_tmp110, consSF);
}

auto tmp111 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp111 at (1085,1-1085,38) */
long double __tmp_in_tmp111;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp111;
tmp111[i0] = ldexp(__tmp_in_tmp111, consSF);
}

auto tmp112 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp112 at (1087,1-1087,38) */
long double __tmp_in_tmp112;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp112;
tmp112[i0] = ldexp(__tmp_in_tmp112, consSF);
}

auto tmp113 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp113 at (1089,1-1089,50) */
long double __tmp_in_tmp113;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
cin >> __tmp_in_tmp113;
tmp113[i0][i1][i2][i3] = ldexp(__tmp_in_tmp113, consSF);
}
}
}
}

auto tmp114 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp114 at (1091,1-1091,49) */
long double __tmp_in_tmp114;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp114;
tmp114[i0][i1][i2][i3] = ldexp(__tmp_in_tmp114, consSF);
}
}
}
}

auto tmp115 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp115 at (1093,1-1093,38) */
long double __tmp_in_tmp115;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp115;
tmp115[i0] = ldexp(__tmp_in_tmp115, consSF);
}

auto tmp116 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp116 at (1095,1-1095,38) */
long double __tmp_in_tmp116;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp116;
tmp116[i0] = ldexp(__tmp_in_tmp116, consSF);
}

auto tmp117 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp117 at (1097,1-1097,38) */
long double __tmp_in_tmp117;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp117;
tmp117[i0] = ldexp(__tmp_in_tmp117, consSF);
}

auto tmp118 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp118 at (1099,1-1099,38) */
long double __tmp_in_tmp118;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp118;
tmp118[i0] = ldexp(__tmp_in_tmp118, consSF);
}

auto tmp119 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp119 at (1101,1-1101,49) */
long double __tmp_in_tmp119;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp119;
tmp119[i0][i1][i2][i3] = ldexp(__tmp_in_tmp119, consSF);
}
}
}
}

auto tmp120 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp120 at (1103,1-1103,38) */
long double __tmp_in_tmp120;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp120;
tmp120[i0] = ldexp(__tmp_in_tmp120, consSF);
}

auto tmp121 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp121 at (1105,1-1105,38) */
long double __tmp_in_tmp121;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp121;
tmp121[i0] = ldexp(__tmp_in_tmp121, consSF);
}

auto tmp122 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp122 at (1107,1-1107,38) */
long double __tmp_in_tmp122;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp122;
tmp122[i0] = ldexp(__tmp_in_tmp122, consSF);
}

auto tmp123 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp123 at (1109,1-1109,38) */
long double __tmp_in_tmp123;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp123;
tmp123[i0] = ldexp(__tmp_in_tmp123, consSF);
}

auto tmp124 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp124 at (1111,1-1111,50) */
long double __tmp_in_tmp124;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
cin >> __tmp_in_tmp124;
tmp124[i0][i1][i2][i3] = ldexp(__tmp_in_tmp124, consSF);
}
}
}
}

auto tmp125 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp125 at (1113,1-1113,39) */
long double __tmp_in_tmp125;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp125;
tmp125[i0] = ldexp(__tmp_in_tmp125, consSF);
}

auto tmp126 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp126 at (1115,1-1115,39) */
long double __tmp_in_tmp126;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp126;
tmp126[i0] = ldexp(__tmp_in_tmp126, consSF);
}

auto tmp127 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp127 at (1117,1-1117,39) */
long double __tmp_in_tmp127;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp127;
tmp127[i0] = ldexp(__tmp_in_tmp127, consSF);
}

auto tmp128 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp128 at (1119,1-1119,39) */
long double __tmp_in_tmp128;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp128;
tmp128[i0] = ldexp(__tmp_in_tmp128, consSF);
}

auto tmp129 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp129 at (1121,1-1121,50) */
long double __tmp_in_tmp129;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp129;
tmp129[i0][i1][i2][i3] = ldexp(__tmp_in_tmp129, consSF);
}
}
}
}

auto tmp130 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp130 at (1123,1-1123,38) */
long double __tmp_in_tmp130;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp130;
tmp130[i0] = ldexp(__tmp_in_tmp130, consSF);
}

auto tmp131 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp131 at (1125,1-1125,38) */
long double __tmp_in_tmp131;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp131;
tmp131[i0] = ldexp(__tmp_in_tmp131, consSF);
}

auto tmp132 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp132 at (1127,1-1127,38) */
long double __tmp_in_tmp132;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp132;
tmp132[i0] = ldexp(__tmp_in_tmp132, consSF);
}

auto tmp133 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp133 at (1129,1-1129,38) */
long double __tmp_in_tmp133;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp133;
tmp133[i0] = ldexp(__tmp_in_tmp133, consSF);
}

auto tmp134 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp134 at (1131,1-1131,49) */
long double __tmp_in_tmp134;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp134;
tmp134[i0][i1][i2][i3] = ldexp(__tmp_in_tmp134, consSF);
}
}
}
}

auto tmp135 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp135 at (1133,1-1133,38) */
long double __tmp_in_tmp135;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp135;
tmp135[i0] = ldexp(__tmp_in_tmp135, consSF);
}

auto tmp136 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp136 at (1135,1-1135,38) */
long double __tmp_in_tmp136;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp136;
tmp136[i0] = ldexp(__tmp_in_tmp136, consSF);
}

auto tmp137 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp137 at (1137,1-1137,38) */
long double __tmp_in_tmp137;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp137;
tmp137[i0] = ldexp(__tmp_in_tmp137, consSF);
}

auto tmp138 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp138 at (1139,1-1139,38) */
long double __tmp_in_tmp138;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp138;
tmp138[i0] = ldexp(__tmp_in_tmp138, consSF);
}

auto tmp139 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp139 at (1141,1-1141,50) */
long double __tmp_in_tmp139;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
cin >> __tmp_in_tmp139;
tmp139[i0][i1][i2][i3] = ldexp(__tmp_in_tmp139, consSF);
}
}
}
}

auto tmp140 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp140 at (1143,1-1143,39) */
long double __tmp_in_tmp140;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp140;
tmp140[i0] = ldexp(__tmp_in_tmp140, consSF);
}

auto tmp141 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp141 at (1145,1-1145,39) */
long double __tmp_in_tmp141;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp141;
tmp141[i0] = ldexp(__tmp_in_tmp141, consSF);
}

auto tmp142 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp142 at (1147,1-1147,39) */
long double __tmp_in_tmp142;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp142;
tmp142[i0] = ldexp(__tmp_in_tmp142, consSF);
}

auto tmp143 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp143 at (1149,1-1149,39) */
long double __tmp_in_tmp143;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp143;
tmp143[i0] = ldexp(__tmp_in_tmp143, consSF);
}

auto tmp144 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp144 at (1151,1-1151,50) */
long double __tmp_in_tmp144;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp144;
tmp144[i0][i1][i2][i3] = ldexp(__tmp_in_tmp144, consSF);
}
}
}
}

auto tmp145 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp145 at (1153,1-1153,38) */
long double __tmp_in_tmp145;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp145;
tmp145[i0] = ldexp(__tmp_in_tmp145, consSF);
}

auto tmp146 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp146 at (1155,1-1155,38) */
long double __tmp_in_tmp146;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp146;
tmp146[i0] = ldexp(__tmp_in_tmp146, consSF);
}

auto tmp147 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp147 at (1157,1-1157,38) */
long double __tmp_in_tmp147;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp147;
tmp147[i0] = ldexp(__tmp_in_tmp147, consSF);
}

auto tmp148 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp148 at (1159,1-1159,38) */
long double __tmp_in_tmp148;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp148;
tmp148[i0] = ldexp(__tmp_in_tmp148, consSF);
}

auto tmp149 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp149 at (1161,1-1161,49) */
long double __tmp_in_tmp149;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp149;
tmp149[i0][i1][i2][i3] = ldexp(__tmp_in_tmp149, consSF);
}
}
}
}

auto tmp150 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp150 at (1163,1-1163,38) */
long double __tmp_in_tmp150;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp150;
tmp150[i0] = ldexp(__tmp_in_tmp150, consSF);
}

auto tmp151 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp151 at (1165,1-1165,38) */
long double __tmp_in_tmp151;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp151;
tmp151[i0] = ldexp(__tmp_in_tmp151, consSF);
}

auto tmp152 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp152 at (1167,1-1167,38) */
long double __tmp_in_tmp152;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp152;
tmp152[i0] = ldexp(__tmp_in_tmp152, consSF);
}

auto tmp153 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp153 at (1169,1-1169,38) */
long double __tmp_in_tmp153;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp153;
tmp153[i0] = ldexp(__tmp_in_tmp153, consSF);
}

auto tmp154 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp154 at (1171,1-1171,50) */
long double __tmp_in_tmp154;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
cin >> __tmp_in_tmp154;
tmp154[i0][i1][i2][i3] = ldexp(__tmp_in_tmp154, consSF);
}
}
}
}

auto tmp155 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp155 at (1173,1-1173,39) */
long double __tmp_in_tmp155;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp155;
tmp155[i0] = ldexp(__tmp_in_tmp155, consSF);
}

auto tmp156 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp156 at (1175,1-1175,39) */
long double __tmp_in_tmp156;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp156;
tmp156[i0] = ldexp(__tmp_in_tmp156, consSF);
}

auto tmp157 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp157 at (1177,1-1177,39) */
long double __tmp_in_tmp157;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp157;
tmp157[i0] = ldexp(__tmp_in_tmp157, consSF);
}

auto tmp158 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp158 at (1179,1-1179,39) */
long double __tmp_in_tmp158;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp158;
tmp158[i0] = ldexp(__tmp_in_tmp158, consSF);
}

auto tmp159 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp159 at (1181,1-1181,50) */
long double __tmp_in_tmp159;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp159;
tmp159[i0][i1][i2][i3] = ldexp(__tmp_in_tmp159, consSF);
}
}
}
}

auto tmp160 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp160 at (1183,1-1183,38) */
long double __tmp_in_tmp160;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp160;
tmp160[i0] = ldexp(__tmp_in_tmp160, consSF);
}

auto tmp161 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp161 at (1185,1-1185,38) */
long double __tmp_in_tmp161;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp161;
tmp161[i0] = ldexp(__tmp_in_tmp161, consSF);
}

auto tmp162 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp162 at (1187,1-1187,38) */
long double __tmp_in_tmp162;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp162;
tmp162[i0] = ldexp(__tmp_in_tmp162, consSF);
}

auto tmp163 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp163 at (1189,1-1189,38) */
long double __tmp_in_tmp163;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp163;
tmp163[i0] = ldexp(__tmp_in_tmp163, consSF);
}

auto tmp164 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp164 at (1191,1-1191,49) */
long double __tmp_in_tmp164;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp164;
tmp164[i0][i1][i2][i3] = ldexp(__tmp_in_tmp164, consSF);
}
}
}
}

auto tmp165 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp165 at (1193,1-1193,38) */
long double __tmp_in_tmp165;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp165;
tmp165[i0] = ldexp(__tmp_in_tmp165, consSF);
}

auto tmp166 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp166 at (1195,1-1195,38) */
long double __tmp_in_tmp166;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp166;
tmp166[i0] = ldexp(__tmp_in_tmp166, consSF);
}

auto tmp167 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp167 at (1197,1-1197,38) */
long double __tmp_in_tmp167;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp167;
tmp167[i0] = ldexp(__tmp_in_tmp167, consSF);
}

auto tmp168 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp168 at (1199,1-1199,38) */
long double __tmp_in_tmp168;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp168;
tmp168[i0] = ldexp(__tmp_in_tmp168, consSF);
}

auto tmp169 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp169 at (1201,1-1201,50) */
long double __tmp_in_tmp169;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
cin >> __tmp_in_tmp169;
tmp169[i0][i1][i2][i3] = ldexp(__tmp_in_tmp169, consSF);
}
}
}
}

auto tmp170 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp170 at (1203,1-1203,39) */
long double __tmp_in_tmp170;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp170;
tmp170[i0] = ldexp(__tmp_in_tmp170, consSF);
}

auto tmp171 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp171 at (1205,1-1205,39) */
long double __tmp_in_tmp171;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp171;
tmp171[i0] = ldexp(__tmp_in_tmp171, consSF);
}

auto tmp172 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp172 at (1207,1-1207,39) */
long double __tmp_in_tmp172;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp172;
tmp172[i0] = ldexp(__tmp_in_tmp172, consSF);
}

auto tmp173 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp173 at (1209,1-1209,39) */
long double __tmp_in_tmp173;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp173;
tmp173[i0] = ldexp(__tmp_in_tmp173, consSF);
}

auto tmp174 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp174 at (1211,1-1211,50) */
long double __tmp_in_tmp174;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp174;
tmp174[i0][i1][i2][i3] = ldexp(__tmp_in_tmp174, consSF);
}
}
}
}

auto tmp175 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp175 at (1213,1-1213,38) */
long double __tmp_in_tmp175;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp175;
tmp175[i0] = ldexp(__tmp_in_tmp175, consSF);
}

auto tmp176 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp176 at (1215,1-1215,38) */
long double __tmp_in_tmp176;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp176;
tmp176[i0] = ldexp(__tmp_in_tmp176, consSF);
}

auto tmp177 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp177 at (1217,1-1217,38) */
long double __tmp_in_tmp177;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp177;
tmp177[i0] = ldexp(__tmp_in_tmp177, consSF);
}

auto tmp178 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp178 at (1219,1-1219,38) */
long double __tmp_in_tmp178;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp178;
tmp178[i0] = ldexp(__tmp_in_tmp178, consSF);
}

auto tmp179 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp179 at (1221,1-1221,49) */
long double __tmp_in_tmp179;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp179;
tmp179[i0][i1][i2][i3] = ldexp(__tmp_in_tmp179, consSF);
}
}
}
}

auto tmp180 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp180 at (1223,1-1223,38) */
long double __tmp_in_tmp180;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp180;
tmp180[i0] = ldexp(__tmp_in_tmp180, consSF);
}

auto tmp181 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp181 at (1225,1-1225,38) */
long double __tmp_in_tmp181;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp181;
tmp181[i0] = ldexp(__tmp_in_tmp181, consSF);
}

auto tmp182 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp182 at (1227,1-1227,38) */
long double __tmp_in_tmp182;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp182;
tmp182[i0] = ldexp(__tmp_in_tmp182, consSF);
}

auto tmp183 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp183 at (1229,1-1229,38) */
long double __tmp_in_tmp183;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp183;
tmp183[i0] = ldexp(__tmp_in_tmp183, consSF);
}

auto tmp184 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp184 at (1231,1-1231,50) */
long double __tmp_in_tmp184;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
cin >> __tmp_in_tmp184;
tmp184[i0][i1][i2][i3] = ldexp(__tmp_in_tmp184, consSF);
}
}
}
}

auto tmp185 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp185 at (1233,1-1233,39) */
long double __tmp_in_tmp185;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp185;
tmp185[i0] = ldexp(__tmp_in_tmp185, consSF);
}

auto tmp186 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp186 at (1235,1-1235,39) */
long double __tmp_in_tmp186;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp186;
tmp186[i0] = ldexp(__tmp_in_tmp186, consSF);
}

auto tmp187 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp187 at (1237,1-1237,39) */
long double __tmp_in_tmp187;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp187;
tmp187[i0] = ldexp(__tmp_in_tmp187, consSF);
}

auto tmp188 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp188 at (1239,1-1239,39) */
long double __tmp_in_tmp188;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp188;
tmp188[i0] = ldexp(__tmp_in_tmp188, consSF);
}

auto tmp189 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp189 at (1241,1-1241,50) */
long double __tmp_in_tmp189;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp189;
tmp189[i0][i1][i2][i3] = ldexp(__tmp_in_tmp189, consSF);
}
}
}
}

auto tmp190 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp190 at (1243,1-1243,38) */
long double __tmp_in_tmp190;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp190;
tmp190[i0] = ldexp(__tmp_in_tmp190, consSF);
}

auto tmp191 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp191 at (1245,1-1245,38) */
long double __tmp_in_tmp191;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp191;
tmp191[i0] = ldexp(__tmp_in_tmp191, consSF);
}

auto tmp192 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp192 at (1247,1-1247,38) */
long double __tmp_in_tmp192;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp192;
tmp192[i0] = ldexp(__tmp_in_tmp192, consSF);
}

auto tmp193 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp193 at (1249,1-1249,38) */
long double __tmp_in_tmp193;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp193;
tmp193[i0] = ldexp(__tmp_in_tmp193, consSF);
}

auto tmp194 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp194 at (1251,1-1251,49) */
long double __tmp_in_tmp194;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp194;
tmp194[i0][i1][i2][i3] = ldexp(__tmp_in_tmp194, consSF);
}
}
}
}

auto tmp195 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp195 at (1253,1-1253,38) */
long double __tmp_in_tmp195;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp195;
tmp195[i0] = ldexp(__tmp_in_tmp195, consSF);
}

auto tmp196 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp196 at (1255,1-1255,38) */
long double __tmp_in_tmp196;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp196;
tmp196[i0] = ldexp(__tmp_in_tmp196, consSF);
}

auto tmp197 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp197 at (1257,1-1257,38) */
long double __tmp_in_tmp197;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp197;
tmp197[i0] = ldexp(__tmp_in_tmp197, consSF);
}

auto tmp198 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp198 at (1259,1-1259,38) */
long double __tmp_in_tmp198;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp198;
tmp198[i0] = ldexp(__tmp_in_tmp198, consSF);
}

auto tmp199 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp199 at (1261,1-1261,50) */
long double __tmp_in_tmp199;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
cin >> __tmp_in_tmp199;
tmp199[i0][i1][i2][i3] = ldexp(__tmp_in_tmp199, consSF);
}
}
}
}

auto tmp200 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp200 at (1263,1-1263,39) */
long double __tmp_in_tmp200;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp200;
tmp200[i0] = ldexp(__tmp_in_tmp200, consSF);
}

auto tmp201 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp201 at (1265,1-1265,39) */
long double __tmp_in_tmp201;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp201;
tmp201[i0] = ldexp(__tmp_in_tmp201, consSF);
}

auto tmp202 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp202 at (1267,1-1267,39) */
long double __tmp_in_tmp202;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp202;
tmp202[i0] = ldexp(__tmp_in_tmp202, consSF);
}

auto tmp203 = make_vector<int32_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp203 at (1269,1-1269,39) */
long double __tmp_in_tmp203;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp203;
tmp203[i0] = ldexp(__tmp_in_tmp203, consSF);
}

auto tmp204 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp204 at (1271,1-1271,51) */
long double __tmp_in_tmp204;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
cin >> __tmp_in_tmp204;
tmp204[i0][i1][i2][i3] = ldexp(__tmp_in_tmp204, consSF);
}
}
}
}

auto tmp205 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp205 at (1273,1-1273,50) */
long double __tmp_in_tmp205;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp205;
tmp205[i0][i1][i2][i3] = ldexp(__tmp_in_tmp205, consSF);
}
}
}
}

auto tmp206 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp206 at (1275,1-1275,38) */
long double __tmp_in_tmp206;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp206;
tmp206[i0] = ldexp(__tmp_in_tmp206, consSF);
}

auto tmp207 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp207 at (1277,1-1277,38) */
long double __tmp_in_tmp207;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp207;
tmp207[i0] = ldexp(__tmp_in_tmp207, consSF);
}

auto tmp208 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp208 at (1279,1-1279,38) */
long double __tmp_in_tmp208;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp208;
tmp208[i0] = ldexp(__tmp_in_tmp208, consSF);
}

auto tmp209 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp209 at (1281,1-1281,38) */
long double __tmp_in_tmp209;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp209;
tmp209[i0] = ldexp(__tmp_in_tmp209, consSF);
}

auto tmp210 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp210 at (1283,1-1283,49) */
long double __tmp_in_tmp210;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp210;
tmp210[i0][i1][i2][i3] = ldexp(__tmp_in_tmp210, consSF);
}
}
}
}

auto tmp211 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp211 at (1285,1-1285,38) */
long double __tmp_in_tmp211;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp211;
tmp211[i0] = ldexp(__tmp_in_tmp211, consSF);
}

auto tmp212 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp212 at (1287,1-1287,38) */
long double __tmp_in_tmp212;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp212;
tmp212[i0] = ldexp(__tmp_in_tmp212, consSF);
}

auto tmp213 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp213 at (1289,1-1289,38) */
long double __tmp_in_tmp213;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp213;
tmp213[i0] = ldexp(__tmp_in_tmp213, consSF);
}

auto tmp214 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp214 at (1291,1-1291,38) */
long double __tmp_in_tmp214;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp214;
tmp214[i0] = ldexp(__tmp_in_tmp214, consSF);
}

auto tmp215 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp215 at (1293,1-1293,50) */
long double __tmp_in_tmp215;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
cin >> __tmp_in_tmp215;
tmp215[i0][i1][i2][i3] = ldexp(__tmp_in_tmp215, consSF);
}
}
}
}

auto tmp216 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp216 at (1295,1-1295,39) */
long double __tmp_in_tmp216;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp216;
tmp216[i0] = ldexp(__tmp_in_tmp216, consSF);
}

auto tmp217 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp217 at (1297,1-1297,39) */
long double __tmp_in_tmp217;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp217;
tmp217[i0] = ldexp(__tmp_in_tmp217, consSF);
}

auto tmp218 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp218 at (1299,1-1299,39) */
long double __tmp_in_tmp218;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp218;
tmp218[i0] = ldexp(__tmp_in_tmp218, consSF);
}

auto tmp219 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp219 at (1301,1-1301,39) */
long double __tmp_in_tmp219;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp219;
tmp219[i0] = ldexp(__tmp_in_tmp219, consSF);
}

auto tmp220 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp220 at (1303,1-1303,50) */
long double __tmp_in_tmp220;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)2048; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp220;
tmp220[i0][i1][i2][i3] = ldexp(__tmp_in_tmp220, consSF);
}
}
}
}

auto tmp221 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp221 at (1305,1-1305,38) */
long double __tmp_in_tmp221;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp221;
tmp221[i0] = ldexp(__tmp_in_tmp221, consSF);
}

auto tmp222 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp222 at (1307,1-1307,38) */
long double __tmp_in_tmp222;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp222;
tmp222[i0] = ldexp(__tmp_in_tmp222, consSF);
}

auto tmp223 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp223 at (1309,1-1309,38) */
long double __tmp_in_tmp223;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp223;
tmp223[i0] = ldexp(__tmp_in_tmp223, consSF);
}

auto tmp224 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp224 at (1311,1-1311,38) */
long double __tmp_in_tmp224;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp224;
tmp224[i0] = ldexp(__tmp_in_tmp224, consSF);
}

auto tmp225 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp225 at (1313,1-1313,49) */
long double __tmp_in_tmp225;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp225;
tmp225[i0][i1][i2][i3] = ldexp(__tmp_in_tmp225, consSF);
}
}
}
}

auto tmp226 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp226 at (1315,1-1315,38) */
long double __tmp_in_tmp226;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp226;
tmp226[i0] = ldexp(__tmp_in_tmp226, consSF);
}

auto tmp227 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp227 at (1317,1-1317,38) */
long double __tmp_in_tmp227;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp227;
tmp227[i0] = ldexp(__tmp_in_tmp227, consSF);
}

auto tmp228 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp228 at (1319,1-1319,38) */
long double __tmp_in_tmp228;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp228;
tmp228[i0] = ldexp(__tmp_in_tmp228, consSF);
}

auto tmp229 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp229 at (1321,1-1321,38) */
long double __tmp_in_tmp229;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp229;
tmp229[i0] = ldexp(__tmp_in_tmp229, consSF);
}

auto tmp230 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp230 at (1323,1-1323,50) */
long double __tmp_in_tmp230;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
cin >> __tmp_in_tmp230;
tmp230[i0][i1][i2][i3] = ldexp(__tmp_in_tmp230, consSF);
}
}
}
}

auto tmp231 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp231 at (1325,1-1325,39) */
long double __tmp_in_tmp231;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp231;
tmp231[i0] = ldexp(__tmp_in_tmp231, consSF);
}

auto tmp232 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp232 at (1327,1-1327,39) */
long double __tmp_in_tmp232;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp232;
tmp232[i0] = ldexp(__tmp_in_tmp232, consSF);
}

auto tmp233 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp233 at (1329,1-1329,39) */
long double __tmp_in_tmp233;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp233;
tmp233[i0] = ldexp(__tmp_in_tmp233, consSF);
}

auto tmp234 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp234 at (1331,1-1331,39) */
long double __tmp_in_tmp234;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp234;
tmp234[i0] = ldexp(__tmp_in_tmp234, consSF);
}

auto tmp235 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp235 at (1333,1-1333,50) */
long double __tmp_in_tmp235;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)2048; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp235;
tmp235[i0][i1][i2][i3] = ldexp(__tmp_in_tmp235, consSF);
}
}
}
}

auto tmp236 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp236 at (1335,1-1335,38) */
long double __tmp_in_tmp236;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp236;
tmp236[i0] = ldexp(__tmp_in_tmp236, consSF);
}

auto tmp237 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp237 at (1337,1-1337,38) */
long double __tmp_in_tmp237;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp237;
tmp237[i0] = ldexp(__tmp_in_tmp237, consSF);
}

auto tmp238 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp238 at (1339,1-1339,38) */
long double __tmp_in_tmp238;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp238;
tmp238[i0] = ldexp(__tmp_in_tmp238, consSF);
}

auto tmp239 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp239 at (1341,1-1341,38) */
long double __tmp_in_tmp239;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp239;
tmp239[i0] = ldexp(__tmp_in_tmp239, consSF);
}

auto tmp240 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp240 at (1343,1-1343,49) */
long double __tmp_in_tmp240;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp240;
tmp240[i0][i1][i2][i3] = ldexp(__tmp_in_tmp240, consSF);
}
}
}
}

auto tmp241 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp241 at (1345,1-1345,38) */
long double __tmp_in_tmp241;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp241;
tmp241[i0] = ldexp(__tmp_in_tmp241, consSF);
}

auto tmp242 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp242 at (1347,1-1347,38) */
long double __tmp_in_tmp242;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp242;
tmp242[i0] = ldexp(__tmp_in_tmp242, consSF);
}

auto tmp243 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp243 at (1349,1-1349,38) */
long double __tmp_in_tmp243;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp243;
tmp243[i0] = ldexp(__tmp_in_tmp243, consSF);
}

auto tmp244 = make_vector<int32_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp244 at (1351,1-1351,38) */
long double __tmp_in_tmp244;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp244;
tmp244[i0] = ldexp(__tmp_in_tmp244, consSF);
}

auto tmp245 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp245 at (1353,1-1353,50) */
long double __tmp_in_tmp245;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
cin >> __tmp_in_tmp245;
tmp245[i0][i1][i2][i3] = ldexp(__tmp_in_tmp245, consSF);
}
}
}
}

auto tmp246 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp246 at (1355,1-1355,39) */
long double __tmp_in_tmp246;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp246;
tmp246[i0] = ldexp(__tmp_in_tmp246, consSF);
}

auto tmp247 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp247 at (1357,1-1357,39) */
long double __tmp_in_tmp247;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp247;
tmp247[i0] = ldexp(__tmp_in_tmp247, consSF);
}

auto tmp248 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp248 at (1359,1-1359,39) */
long double __tmp_in_tmp248;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp248;
tmp248[i0] = ldexp(__tmp_in_tmp248, consSF);
}

auto tmp249 = make_vector<int32_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp249 at (1361,1-1361,39) */
long double __tmp_in_tmp249;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
cin >> __tmp_in_tmp249;
tmp249[i0] = ldexp(__tmp_in_tmp249, consSF);
}

auto tmp250 = make_vector<int32_t>( (int32_t)2048,  (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp250 at (1363,1-1363,45) */
long double __tmp_in_tmp250;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1001; i1++){
cin >> __tmp_in_tmp250;
tmp250[i0][i1] = ldexp(__tmp_in_tmp250, consSF);
}
}

auto tmp251 = make_vector<int32_t>( (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp251 at (1365,1-1365,39) */
long double __tmp_in_tmp251;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1001; i0++){
cin >> __tmp_in_tmp251;
tmp251[i0] = ldexp(__tmp_in_tmp251, consSF);
}

auto tmp0 = make_vector<int32_t>( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3);

int randomSubsetAcutalImgIdxArr[randomSubsetNumImages];
bool choosingImgFromRandomSubset = (randomSubsetIdxTestFileName!="");
if (choosingImgFromRandomSubset){
	readIdxFromRandomSubsetFile(randomSubsetIdxTestFileName, randomSubsetNumImages, randomSubsetAcutalImgIdxArr);
}

for(int __imgCounter = startImgNum; __imgCounter < endImgNum; __imgCounter++){
	cout<<"Answer for image number = "<<__imgCounter<<":"<<endl;

	int actualIdx = __imgCounter;
	if (choosingImgFromRandomSubset){
		actualIdx = randomSubsetAcutalImgIdxArr[__imgCounter-1];
	}
	
	/* Variable to read the clear value corresponding to the input variable tmp0 at (863,1-863,47) */
	long double __tmp_in_tmp0;
	string line;
	string inputImgFileName = preProcessedImgDir + "/ImageNum_" + to_string(actualIdx) + ".inp";
	ifstream myfile(inputImgFileName);
	getline(myfile, line);
	stringstream lineStream(line);
	string num;

	for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)224; i1++){
	for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)224; i2++){
	for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)3; i3++){
	lineStream >> num;
	__tmp_in_tmp0 = stold(num);
	tmp0[i0][i1][i2][i3] = ldexp(__tmp_in_tmp0, consSF);
	}
	}
	}
	}


	tmp252[ (int32_t)0][ (int32_t)0] =  (int32_t)0;
	tmp252[ (int32_t)0][ (int32_t)1] =  (int32_t)0;
	tmp252[ (int32_t)1][ (int32_t)0] =  (int32_t)3;
	tmp252[ (int32_t)1][ (int32_t)1] =  (int32_t)3;
	tmp252[ (int32_t)2][ (int32_t)0] =  (int32_t)3;
	tmp252[ (int32_t)2][ (int32_t)1] =  (int32_t)3;
	tmp252[ (int32_t)3][ (int32_t)0] =  (int32_t)0;
	tmp252[ (int32_t)3][ (int32_t)1] =  (int32_t)0;
	Pad442( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0,  (int32_t)4,  (int32_t)2, tmp252, tmp253);
	Conv2DCSF( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)7,  (int32_t)7,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp253, tmp1, tmp254,  consSF);
	MaxPool( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)1,  (int32_t)0,  (int32_t)1,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp254, tmp255);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp255,  (int32_t)64, tmp2, tmp3, tmp256, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp256, tmp257);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp257, tmp6, tmp258,  consSF);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp257, tmp7, tmp259,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp259,  (int32_t)64, tmp8, tmp9, tmp260, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp260, tmp261);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp261, tmp12, tmp262,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp262,  (int32_t)64, tmp13, tmp14, tmp263, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp263, tmp264);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp264, tmp17, tmp265,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp265, tmp258, tmp266);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp266,  (int32_t)256, tmp18, tmp19, tmp267, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp267, tmp268);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp268, tmp22, tmp269,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp269,  (int32_t)64, tmp23, tmp24, tmp270, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp270, tmp271);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp271, tmp27, tmp272,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp272,  (int32_t)64, tmp28, tmp29, tmp273, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp273, tmp274);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp274, tmp32, tmp275,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp275, tmp266, tmp276);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp276,  (int32_t)256, tmp33, tmp34, tmp277, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp277, tmp278);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp278, tmp37, tmp279,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp279,  (int32_t)64, tmp38, tmp39, tmp280, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp280, tmp281);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp281, tmp42, tmp282,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp282,  (int32_t)64, tmp43, tmp44, tmp283, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp283, tmp284);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp284, tmp47, tmp285,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp285, tmp276, tmp286);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp286,  (int32_t)256, tmp48, tmp49, tmp287, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp287, tmp288);
	tmp289[ (int32_t)0][ (int32_t)0] =  (int32_t)0;
	tmp289[ (int32_t)0][ (int32_t)1] =  (int32_t)0;
	tmp289[ (int32_t)1][ (int32_t)0] =  (int32_t)0;
	tmp289[ (int32_t)1][ (int32_t)1] =  (int32_t)0;
	tmp289[ (int32_t)2][ (int32_t)0] =  (int32_t)0;
	tmp289[ (int32_t)2][ (int32_t)1] =  (int32_t)0;
	tmp289[ (int32_t)3][ (int32_t)0] =  (int32_t)0;
	tmp289[ (int32_t)3][ (int32_t)1] =  (int32_t)0;
	Pad442( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp288,  (int32_t)4,  (int32_t)2, tmp289, tmp290);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp290, tmp52, tmp291,  consSF);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp288, tmp53, tmp292,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp292,  (int32_t)128, tmp54, tmp55, tmp293, consSF);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp293, tmp294);
	tmp295[ (int32_t)0][ (int32_t)0] =  (int32_t)0;
	tmp295[ (int32_t)0][ (int32_t)1] =  (int32_t)0;
	tmp295[ (int32_t)1][ (int32_t)0] =  (int32_t)1;
	tmp295[ (int32_t)1][ (int32_t)1] =  (int32_t)1;
	tmp295[ (int32_t)2][ (int32_t)0] =  (int32_t)1;
	tmp295[ (int32_t)2][ (int32_t)1] =  (int32_t)1;
	tmp295[ (int32_t)3][ (int32_t)0] =  (int32_t)0;
	tmp295[ (int32_t)3][ (int32_t)1] =  (int32_t)0;
	Pad442( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp294,  (int32_t)4,  (int32_t)2, tmp295, tmp296);
	Conv2DCSF( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp296, tmp58, tmp297,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp297,  (int32_t)128, tmp59, tmp60, tmp298, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp298, tmp299);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp299, tmp63, tmp300,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp300, tmp291, tmp301);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp301,  (int32_t)512, tmp64, tmp65, tmp302, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp302, tmp303);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp303, tmp68, tmp304,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp304,  (int32_t)128, tmp69, tmp70, tmp305, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp305, tmp306);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp306, tmp73, tmp307,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp307,  (int32_t)128, tmp74, tmp75, tmp308, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp308, tmp309);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp309, tmp78, tmp310,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp310, tmp301, tmp311);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp311,  (int32_t)512, tmp79, tmp80, tmp312, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp312, tmp313);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp313, tmp83, tmp314,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp314,  (int32_t)128, tmp84, tmp85, tmp315, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp315, tmp316);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp316, tmp88, tmp317,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp317,  (int32_t)128, tmp89, tmp90, tmp318, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp318, tmp319);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp319, tmp93, tmp320,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp320, tmp311, tmp321);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp321,  (int32_t)512, tmp94, tmp95, tmp322, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp322, tmp323);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp323, tmp98, tmp324,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp324,  (int32_t)128, tmp99, tmp100, tmp325, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp325, tmp326);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp326, tmp103, tmp327,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp327,  (int32_t)128, tmp104, tmp105, tmp328, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp328, tmp329);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp329, tmp108, tmp330,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp330, tmp321, tmp331);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp331,  (int32_t)512, tmp109, tmp110, tmp332, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp332, tmp333);
	tmp334[ (int32_t)0][ (int32_t)0] =  (int32_t)0;
	tmp334[ (int32_t)0][ (int32_t)1] =  (int32_t)0;
	tmp334[ (int32_t)1][ (int32_t)0] =  (int32_t)0;
	tmp334[ (int32_t)1][ (int32_t)1] =  (int32_t)0;
	tmp334[ (int32_t)2][ (int32_t)0] =  (int32_t)0;
	tmp334[ (int32_t)2][ (int32_t)1] =  (int32_t)0;
	tmp334[ (int32_t)3][ (int32_t)0] =  (int32_t)0;
	tmp334[ (int32_t)3][ (int32_t)1] =  (int32_t)0;
	Pad442( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp333,  (int32_t)4,  (int32_t)2, tmp334, tmp335);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp335, tmp113, tmp336,  consSF);
	Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp333, tmp114, tmp337,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp337,  (int32_t)256, tmp115, tmp116, tmp338, consSF);
	Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp338, tmp339);
	tmp340[ (int32_t)0][ (int32_t)0] =  (int32_t)0;
	tmp340[ (int32_t)0][ (int32_t)1] =  (int32_t)0;
	tmp340[ (int32_t)1][ (int32_t)0] =  (int32_t)1;
	tmp340[ (int32_t)1][ (int32_t)1] =  (int32_t)1;
	tmp340[ (int32_t)2][ (int32_t)0] =  (int32_t)1;
	tmp340[ (int32_t)2][ (int32_t)1] =  (int32_t)1;
	tmp340[ (int32_t)3][ (int32_t)0] =  (int32_t)0;
	tmp340[ (int32_t)3][ (int32_t)1] =  (int32_t)0;
	Pad442( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp339,  (int32_t)4,  (int32_t)2, tmp340, tmp341);
	Conv2DCSF( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp341, tmp119, tmp342,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp342,  (int32_t)256, tmp120, tmp121, tmp343, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp343, tmp344);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp344, tmp124, tmp345,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp345, tmp336, tmp346);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp346,  (int32_t)1024, tmp125, tmp126, tmp347, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp347, tmp348);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp348, tmp129, tmp349,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp349,  (int32_t)256, tmp130, tmp131, tmp350, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp350, tmp351);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp351, tmp134, tmp352,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp352,  (int32_t)256, tmp135, tmp136, tmp353, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp353, tmp354);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp354, tmp139, tmp355,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp355, tmp346, tmp356);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp356,  (int32_t)1024, tmp140, tmp141, tmp357, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp357, tmp358);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp358, tmp144, tmp359,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp359,  (int32_t)256, tmp145, tmp146, tmp360, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp360, tmp361);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp361, tmp149, tmp362,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp362,  (int32_t)256, tmp150, tmp151, tmp363, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp363, tmp364);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp364, tmp154, tmp365,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp365, tmp356, tmp366);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp366,  (int32_t)1024, tmp155, tmp156, tmp367, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp367, tmp368);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp368, tmp159, tmp369,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp369,  (int32_t)256, tmp160, tmp161, tmp370, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp370, tmp371);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp371, tmp164, tmp372,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp372,  (int32_t)256, tmp165, tmp166, tmp373, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp373, tmp374);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp374, tmp169, tmp375,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp375, tmp366, tmp376);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp376,  (int32_t)1024, tmp170, tmp171, tmp377, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp377, tmp378);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp378, tmp174, tmp379,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp379,  (int32_t)256, tmp175, tmp176, tmp380, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp380, tmp381);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp381, tmp179, tmp382,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp382,  (int32_t)256, tmp180, tmp181, tmp383, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp383, tmp384);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp384, tmp184, tmp385,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp385, tmp376, tmp386);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp386,  (int32_t)1024, tmp185, tmp186, tmp387, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp387, tmp388);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp388, tmp189, tmp389,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp389,  (int32_t)256, tmp190, tmp191, tmp390, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp390, tmp391);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp391, tmp194, tmp392,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp392,  (int32_t)256, tmp195, tmp196, tmp393, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp393, tmp394);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp394, tmp199, tmp395,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp395, tmp386, tmp396);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp396,  (int32_t)1024, tmp200, tmp201, tmp397, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp397, tmp398);
	tmp399[ (int32_t)0][ (int32_t)0] =  (int32_t)0;
	tmp399[ (int32_t)0][ (int32_t)1] =  (int32_t)0;
	tmp399[ (int32_t)1][ (int32_t)0] =  (int32_t)0;
	tmp399[ (int32_t)1][ (int32_t)1] =  (int32_t)0;
	tmp399[ (int32_t)2][ (int32_t)0] =  (int32_t)0;
	tmp399[ (int32_t)2][ (int32_t)1] =  (int32_t)0;
	tmp399[ (int32_t)3][ (int32_t)0] =  (int32_t)0;
	tmp399[ (int32_t)3][ (int32_t)1] =  (int32_t)0;
	Pad442( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp398,  (int32_t)4,  (int32_t)2, tmp399, tmp400);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp400, tmp204, tmp401,  consSF);
	Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp398, tmp205, tmp402,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp402,  (int32_t)512, tmp206, tmp207, tmp403, consSF);
	Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp403, tmp404);
	tmp405[ (int32_t)0][ (int32_t)0] =  (int32_t)0;
	tmp405[ (int32_t)0][ (int32_t)1] =  (int32_t)0;
	tmp405[ (int32_t)1][ (int32_t)0] =  (int32_t)1;
	tmp405[ (int32_t)1][ (int32_t)1] =  (int32_t)1;
	tmp405[ (int32_t)2][ (int32_t)0] =  (int32_t)1;
	tmp405[ (int32_t)2][ (int32_t)1] =  (int32_t)1;
	tmp405[ (int32_t)3][ (int32_t)0] =  (int32_t)0;
	tmp405[ (int32_t)3][ (int32_t)1] =  (int32_t)0;
	Pad442( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp404,  (int32_t)4,  (int32_t)2, tmp405, tmp406);
	Conv2DCSF( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp406, tmp210, tmp407,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp407,  (int32_t)512, tmp211, tmp212, tmp408, consSF);
	Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp408, tmp409);
	Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp409, tmp215, tmp410,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp410, tmp401, tmp411);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp411,  (int32_t)2048, tmp216, tmp217, tmp412, consSF);
	Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp412, tmp413);
	Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp413, tmp220, tmp414,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp414,  (int32_t)512, tmp221, tmp222, tmp415, consSF);
	Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp415, tmp416);
	Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp416, tmp225, tmp417,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp417,  (int32_t)512, tmp226, tmp227, tmp418, consSF);
	Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp418, tmp419);
	Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp419, tmp230, tmp420,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp420, tmp411, tmp421);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp421,  (int32_t)2048, tmp231, tmp232, tmp422, consSF);
	Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp422, tmp423);
	Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp423, tmp235, tmp424,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp424,  (int32_t)512, tmp236, tmp237, tmp425, consSF);
	Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp425, tmp426);
	Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp426, tmp240, tmp427,  consSF);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp427,  (int32_t)512, tmp241, tmp242, tmp428, consSF);
	Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp428, tmp429);
	Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp429, tmp245, tmp430,  consSF);
	MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp430, tmp421, tmp431);
	TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp431,  (int32_t)2048, tmp246, tmp247, tmp432, consSF);
	Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp432, tmp433);
	AvgPool( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)7,  (int32_t)7,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp433, tmp434);
	Squeeze24( (int32_t)1,  (int32_t)2048,  (int32_t)1,  (int32_t)2,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048, tmp434, tmp435);
	MatMulCSF2D( (int32_t)1,  (int32_t)2048,  (int32_t)1001, tmp435, tmp250, tmp436,  consSF);
	MatAddBroadCast2( (int32_t)1,  (int32_t)1001, tmp436, tmp251, tmp437);
	// ArgMax1( (int32_t)1,  (int32_t)1,  (int32_t)1001, tmp437,  (int32_t)1, tmp438);
	for(int i=0;i<1001;i++){
		cout<<tmp437[0][i]<<" ";
	}
	cout<<endl;
}
return 0;
}

