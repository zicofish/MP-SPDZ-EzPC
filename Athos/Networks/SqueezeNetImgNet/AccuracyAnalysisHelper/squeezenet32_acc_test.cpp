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
	Matrix<int32_t, Dynamic, Dynamic> eigen_a(i, j);
	Matrix<int32_t, Dynamic, Dynamic> eigen_b(j, k);
	Matrix<int32_t, Dynamic, Dynamic> eigen_c(i, k);

	for (int i0 = 0; i0 < i; ++i0){
		for (int i1 = 0; i1 < j; ++i1){
			eigen_a(i0, i1) = A[i0][i1];
		}
	}

	for (int i0 = 0; i0 < j; ++i0){
		for (int i1 = 0; i1 < k; ++i1){
			eigen_b(i0, i1) = B[i0][i1];
		}
	}

	eigen_c = eigen_a * eigen_b;

	for (int i0 = 0; i0 < i; ++i0){
		for (int i1 = 0; i1 < k; ++i1){
			C[i0][i1] = (eigen_c(i0, i1) >> consSF);
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

auto tmp53 = make_vector<int32_t>( (int32_t)1,  (int32_t)113,  (int32_t)113,  (int32_t)64);

auto tmp54 = make_vector<int32_t>( (int32_t)1,  (int32_t)113,  (int32_t)113,  (int32_t)64);

auto tmp55 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp56 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp57 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16);

auto tmp58 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16);

auto tmp59 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16);

auto tmp60 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp61 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp62 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp63 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp64 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp65 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp66 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp67 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16);

auto tmp68 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16);

auto tmp69 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16);

auto tmp70 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp71 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp72 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp73 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp74 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp75 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp76 = make_vector<int32_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp77 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp78 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32);

auto tmp79 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32);

auto tmp80 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32);

auto tmp81 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp82 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp83 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp84 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp85 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp86 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp87 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256);

auto tmp88 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32);

auto tmp89 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32);

auto tmp90 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32);

auto tmp91 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp92 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp93 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp94 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp95 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp96 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128);

auto tmp97 = make_vector<int32_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256);

auto tmp98 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp99 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48);

auto tmp100 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48);

auto tmp101 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48);

auto tmp102 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp103 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp104 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp105 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp106 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp107 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp108 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384);

auto tmp109 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48);

auto tmp110 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48);

auto tmp111 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48);

auto tmp112 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp113 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp114 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp115 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp116 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp117 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192);

auto tmp118 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384);

auto tmp119 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64);

auto tmp120 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64);

auto tmp121 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64);

auto tmp122 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp123 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp124 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp125 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp126 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp127 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp128 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512);

auto tmp129 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64);

auto tmp130 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64);

auto tmp131 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64);

auto tmp132 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp133 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp134 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp135 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp136 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp137 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256);

auto tmp138 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512);

auto tmp139 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000);

auto tmp140 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000);

auto tmp141 = make_vector<int32_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000);

auto tmp142 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000);

auto tmp1 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)3,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp1 at (768,1-768,44) */
long double __tmp_in_tmp1;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)3; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp1;
tmp1[i0][i1][i2][i3] = ldexp(__tmp_in_tmp1, consSF);
}
}
}
}

auto tmp2 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp2 at (770,1-770,35) */
long double __tmp_in_tmp2;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp2;
tmp2[i0] = ldexp(__tmp_in_tmp2, consSF);
}

auto tmp3 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)16);
/* Variable to read the clear value corresponding to the input variable tmp3 at (772,1-772,45) */
long double __tmp_in_tmp3;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)16; i3++){
cin >> __tmp_in_tmp3;
tmp3[i0][i1][i2][i3] = ldexp(__tmp_in_tmp3, consSF);
}
}
}
}

auto tmp4 = make_vector<int32_t>( (int32_t)16);
/* Variable to read the clear value corresponding to the input variable tmp4 at (774,1-774,35) */
long double __tmp_in_tmp4;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)16; i0++){
cin >> __tmp_in_tmp4;
tmp4[i0] = ldexp(__tmp_in_tmp4, consSF);
}

auto tmp5 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp5 at (776,1-776,45) */
long double __tmp_in_tmp5;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)16; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp5;
tmp5[i0][i1][i2][i3] = ldexp(__tmp_in_tmp5, consSF);
}
}
}
}

auto tmp6 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp6 at (778,1-778,35) */
long double __tmp_in_tmp6;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp6;
tmp6[i0] = ldexp(__tmp_in_tmp6, consSF);
}

auto tmp7 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)16,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp7 at (780,1-780,45) */
long double __tmp_in_tmp7;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)16; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp7;
tmp7[i0][i1][i2][i3] = ldexp(__tmp_in_tmp7, consSF);
}
}
}
}

auto tmp8 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp8 at (782,1-782,35) */
long double __tmp_in_tmp8;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp8;
tmp8[i0] = ldexp(__tmp_in_tmp8, consSF);
}

auto tmp9 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)16);
/* Variable to read the clear value corresponding to the input variable tmp9 at (784,1-784,46) */
long double __tmp_in_tmp9;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)16; i3++){
cin >> __tmp_in_tmp9;
tmp9[i0][i1][i2][i3] = ldexp(__tmp_in_tmp9, consSF);
}
}
}
}

auto tmp10 = make_vector<int32_t>( (int32_t)16);
/* Variable to read the clear value corresponding to the input variable tmp10 at (786,1-786,36) */
long double __tmp_in_tmp10;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)16; i0++){
cin >> __tmp_in_tmp10;
tmp10[i0] = ldexp(__tmp_in_tmp10, consSF);
}

auto tmp11 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp11 at (788,1-788,46) */
long double __tmp_in_tmp11;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)16; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp11;
tmp11[i0][i1][i2][i3] = ldexp(__tmp_in_tmp11, consSF);
}
}
}
}

auto tmp12 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp12 at (790,1-790,36) */
long double __tmp_in_tmp12;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp12;
tmp12[i0] = ldexp(__tmp_in_tmp12, consSF);
}

auto tmp13 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)16,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp13 at (792,1-792,46) */
long double __tmp_in_tmp13;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)16; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp13;
tmp13[i0][i1][i2][i3] = ldexp(__tmp_in_tmp13, consSF);
}
}
}
}

auto tmp14 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp14 at (794,1-794,36) */
long double __tmp_in_tmp14;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp14;
tmp14[i0] = ldexp(__tmp_in_tmp14, consSF);
}

auto tmp15 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp15 at (796,1-796,47) */
long double __tmp_in_tmp15;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp15;
tmp15[i0][i1][i2][i3] = ldexp(__tmp_in_tmp15, consSF);
}
}
}
}

auto tmp16 = make_vector<int32_t>( (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp16 at (798,1-798,36) */
long double __tmp_in_tmp16;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)32; i0++){
cin >> __tmp_in_tmp16;
tmp16[i0] = ldexp(__tmp_in_tmp16, consSF);
}

auto tmp17 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)32,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp17 at (800,1-800,47) */
long double __tmp_in_tmp17;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)32; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp17;
tmp17[i0][i1][i2][i3] = ldexp(__tmp_in_tmp17, consSF);
}
}
}
}

auto tmp18 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp18 at (802,1-802,37) */
long double __tmp_in_tmp18;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp18;
tmp18[i0] = ldexp(__tmp_in_tmp18, consSF);
}

auto tmp19 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp19 at (804,1-804,47) */
long double __tmp_in_tmp19;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)32; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp19;
tmp19[i0][i1][i2][i3] = ldexp(__tmp_in_tmp19, consSF);
}
}
}
}

auto tmp20 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp20 at (806,1-806,37) */
long double __tmp_in_tmp20;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp20;
tmp20[i0] = ldexp(__tmp_in_tmp20, consSF);
}

auto tmp21 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp21 at (808,1-808,47) */
long double __tmp_in_tmp21;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp21;
tmp21[i0][i1][i2][i3] = ldexp(__tmp_in_tmp21, consSF);
}
}
}
}

auto tmp22 = make_vector<int32_t>( (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp22 at (810,1-810,36) */
long double __tmp_in_tmp22;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)32; i0++){
cin >> __tmp_in_tmp22;
tmp22[i0] = ldexp(__tmp_in_tmp22, consSF);
}

auto tmp23 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)32,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp23 at (812,1-812,47) */
long double __tmp_in_tmp23;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)32; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp23;
tmp23[i0][i1][i2][i3] = ldexp(__tmp_in_tmp23, consSF);
}
}
}
}

auto tmp24 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp24 at (814,1-814,37) */
long double __tmp_in_tmp24;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp24;
tmp24[i0] = ldexp(__tmp_in_tmp24, consSF);
}

auto tmp25 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp25 at (816,1-816,47) */
long double __tmp_in_tmp25;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)32; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp25;
tmp25[i0][i1][i2][i3] = ldexp(__tmp_in_tmp25, consSF);
}
}
}
}

auto tmp26 = make_vector<int32_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp26 at (818,1-818,37) */
long double __tmp_in_tmp26;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp26;
tmp26[i0] = ldexp(__tmp_in_tmp26, consSF);
}

auto tmp27 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)48);
/* Variable to read the clear value corresponding to the input variable tmp27 at (820,1-820,47) */
long double __tmp_in_tmp27;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)48; i3++){
cin >> __tmp_in_tmp27;
tmp27[i0][i1][i2][i3] = ldexp(__tmp_in_tmp27, consSF);
}
}
}
}

auto tmp28 = make_vector<int32_t>( (int32_t)48);
/* Variable to read the clear value corresponding to the input variable tmp28 at (822,1-822,36) */
long double __tmp_in_tmp28;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)48; i0++){
cin >> __tmp_in_tmp28;
tmp28[i0] = ldexp(__tmp_in_tmp28, consSF);
}

auto tmp29 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)48,  (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp29 at (824,1-824,47) */
long double __tmp_in_tmp29;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)48; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)192; i3++){
cin >> __tmp_in_tmp29;
tmp29[i0][i1][i2][i3] = ldexp(__tmp_in_tmp29, consSF);
}
}
}
}

auto tmp30 = make_vector<int32_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp30 at (826,1-826,37) */
long double __tmp_in_tmp30;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp30;
tmp30[i0] = ldexp(__tmp_in_tmp30, consSF);
}

auto tmp31 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)48,  (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp31 at (828,1-828,47) */
long double __tmp_in_tmp31;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)48; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)192; i3++){
cin >> __tmp_in_tmp31;
tmp31[i0][i1][i2][i3] = ldexp(__tmp_in_tmp31, consSF);
}
}
}
}

auto tmp32 = make_vector<int32_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp32 at (830,1-830,37) */
long double __tmp_in_tmp32;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp32;
tmp32[i0] = ldexp(__tmp_in_tmp32, consSF);
}

auto tmp33 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)48);
/* Variable to read the clear value corresponding to the input variable tmp33 at (832,1-832,47) */
long double __tmp_in_tmp33;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)384; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)48; i3++){
cin >> __tmp_in_tmp33;
tmp33[i0][i1][i2][i3] = ldexp(__tmp_in_tmp33, consSF);
}
}
}
}

auto tmp34 = make_vector<int32_t>( (int32_t)48);
/* Variable to read the clear value corresponding to the input variable tmp34 at (834,1-834,36) */
long double __tmp_in_tmp34;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)48; i0++){
cin >> __tmp_in_tmp34;
tmp34[i0] = ldexp(__tmp_in_tmp34, consSF);
}

auto tmp35 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)48,  (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp35 at (836,1-836,47) */
long double __tmp_in_tmp35;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)48; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)192; i3++){
cin >> __tmp_in_tmp35;
tmp35[i0][i1][i2][i3] = ldexp(__tmp_in_tmp35, consSF);
}
}
}
}

auto tmp36 = make_vector<int32_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp36 at (838,1-838,37) */
long double __tmp_in_tmp36;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp36;
tmp36[i0] = ldexp(__tmp_in_tmp36, consSF);
}

auto tmp37 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)48,  (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp37 at (840,1-840,47) */
long double __tmp_in_tmp37;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)48; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)192; i3++){
cin >> __tmp_in_tmp37;
tmp37[i0][i1][i2][i3] = ldexp(__tmp_in_tmp37, consSF);
}
}
}
}

auto tmp38 = make_vector<int32_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp38 at (842,1-842,37) */
long double __tmp_in_tmp38;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp38;
tmp38[i0] = ldexp(__tmp_in_tmp38, consSF);
}

auto tmp39 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp39 at (844,1-844,47) */
long double __tmp_in_tmp39;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)384; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp39;
tmp39[i0][i1][i2][i3] = ldexp(__tmp_in_tmp39, consSF);
}
}
}
}

auto tmp40 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp40 at (846,1-846,36) */
long double __tmp_in_tmp40;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp40;
tmp40[i0] = ldexp(__tmp_in_tmp40, consSF);
}

auto tmp41 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp41 at (848,1-848,47) */
long double __tmp_in_tmp41;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp41;
tmp41[i0][i1][i2][i3] = ldexp(__tmp_in_tmp41, consSF);
}
}
}
}

auto tmp42 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp42 at (850,1-850,37) */
long double __tmp_in_tmp42;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp42;
tmp42[i0] = ldexp(__tmp_in_tmp42, consSF);
}

auto tmp43 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp43 at (852,1-852,47) */
long double __tmp_in_tmp43;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp43;
tmp43[i0][i1][i2][i3] = ldexp(__tmp_in_tmp43, consSF);
}
}
}
}

auto tmp44 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp44 at (854,1-854,37) */
long double __tmp_in_tmp44;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp44;
tmp44[i0] = ldexp(__tmp_in_tmp44, consSF);
}

auto tmp45 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp45 at (856,1-856,47) */
long double __tmp_in_tmp45;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp45;
tmp45[i0][i1][i2][i3] = ldexp(__tmp_in_tmp45, consSF);
}
}
}
}

auto tmp46 = make_vector<int32_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp46 at (858,1-858,36) */
long double __tmp_in_tmp46;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp46;
tmp46[i0] = ldexp(__tmp_in_tmp46, consSF);
}

auto tmp47 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp47 at (860,1-860,47) */
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
/* Variable to read the clear value corresponding to the input variable tmp48 at (862,1-862,37) */
long double __tmp_in_tmp48;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp48;
tmp48[i0] = ldexp(__tmp_in_tmp48, consSF);
}

auto tmp49 = make_vector<int32_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp49 at (864,1-864,47) */
long double __tmp_in_tmp49;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp49;
tmp49[i0][i1][i2][i3] = ldexp(__tmp_in_tmp49, consSF);
}
}
}
}

auto tmp50 = make_vector<int32_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp50 at (866,1-866,37) */
long double __tmp_in_tmp50;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp50;
tmp50[i0] = ldexp(__tmp_in_tmp50, consSF);
}

auto tmp51 = make_vector<int32_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1000);
/* Variable to read the clear value corresponding to the input variable tmp51 at (868,1-868,49) */
long double __tmp_in_tmp51;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1000; i3++){
cin >> __tmp_in_tmp51;
tmp51[i0][i1][i2][i3] = ldexp(__tmp_in_tmp51, consSF);
}
}
}
}

auto tmp52 = make_vector<int32_t>( (int32_t)1000);
/* Variable to read the clear value corresponding to the input variable tmp52 at (870,1-870,38) */
long double __tmp_in_tmp52;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1000; i0++){
cin >> __tmp_in_tmp52;
tmp52[i0] = ldexp(__tmp_in_tmp52, consSF);
}


auto tmp0 = make_vector<int32_t>( (int32_t)1,  (int32_t)227,  (int32_t)227,  (int32_t)3);

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
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)227; i1++){
	for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)227; i2++){
	for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)3; i3++){
	lineStream >> num;
	__tmp_in_tmp0 = stold(num);
	tmp0[i0][i1][i2][i3] = ldexp(__tmp_in_tmp0, consSF);
	}
	}
	}
	}

	Conv2DCSF( (int32_t)1,  (int32_t)227,  (int32_t)227,  (int32_t)3,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp0, tmp1, tmp53,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)113,  (int32_t)113,  (int32_t)64, tmp53, tmp2, tmp54);
	MaxPool( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)113,  (int32_t)113,  (int32_t)64, tmp54, tmp55);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp55, tmp56);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp56, tmp3, tmp57,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp57, tmp4, tmp58);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp58, tmp59);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp59, tmp5, tmp60,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp60, tmp6, tmp61);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp61, tmp62);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp59, tmp7, tmp63,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp63, tmp8, tmp64);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp64, tmp65);
	Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp62,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp65,  (int32_t)3, tmp66);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp66, tmp9, tmp67,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp67, tmp10, tmp68);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp68, tmp69);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp69, tmp11, tmp70,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp70, tmp12, tmp71);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp71, tmp72);
	Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp69, tmp13, tmp73,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp73, tmp14, tmp74);
	Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp74, tmp75);
	Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp72,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp75,  (int32_t)3, tmp76);
	MaxPool( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp76, tmp77);
	Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)32,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp77, tmp15, tmp78,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp78, tmp16, tmp79);
	Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp79, tmp80);
	Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp80, tmp17, tmp81,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp81, tmp18, tmp82);
	Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp82, tmp83);
	Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp80, tmp19, tmp84,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp84, tmp20, tmp85);
	Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp85, tmp86);
	Concat2T444( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256,  (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp83,  (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp86,  (int32_t)3, tmp87);
	Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)32,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp87, tmp21, tmp88,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp88, tmp22, tmp89);
	Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp89, tmp90);
	Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp90, tmp23, tmp91,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp91, tmp24, tmp92);
	Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp92, tmp93);
	Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp90, tmp25, tmp94,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp94, tmp26, tmp95);
	Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp95, tmp96);
	Concat2T444( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256,  (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp93,  (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp96,  (int32_t)3, tmp97);
	MaxPool( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256, tmp97, tmp98);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)48,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp98, tmp27, tmp99,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp99, tmp28, tmp100);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp100, tmp101);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48,  (int32_t)1,  (int32_t)1,  (int32_t)192,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp101, tmp29, tmp102,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp102, tmp30, tmp103);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp103, tmp104);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48,  (int32_t)3,  (int32_t)3,  (int32_t)192,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp101, tmp31, tmp105,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp105, tmp32, tmp106);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp106, tmp107);
	Concat2T444( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp104,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp107,  (int32_t)3, tmp108);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384,  (int32_t)1,  (int32_t)1,  (int32_t)48,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp108, tmp33, tmp109,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp109, tmp34, tmp110);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp110, tmp111);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48,  (int32_t)1,  (int32_t)1,  (int32_t)192,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp111, tmp35, tmp112,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp112, tmp36, tmp113);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp113, tmp114);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48,  (int32_t)3,  (int32_t)3,  (int32_t)192,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp111, tmp37, tmp115,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp115, tmp38, tmp116);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp116, tmp117);
	Concat2T444( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp114,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp117,  (int32_t)3, tmp118);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp118, tmp39, tmp119,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp119, tmp40, tmp120);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp120, tmp121);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp121, tmp41, tmp122,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp122, tmp42, tmp123);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp123, tmp124);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp121, tmp43, tmp125,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp125, tmp44, tmp126);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp126, tmp127);
	Concat2T444( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp124,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp127,  (int32_t)3, tmp128);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp128, tmp45, tmp129,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp129, tmp46, tmp130);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp130, tmp131);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp131, tmp47, tmp132,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp132, tmp48, tmp133);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp133, tmp134);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp131, tmp49, tmp135,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp135, tmp50, tmp136);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp136, tmp137);
	Concat2T444( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp134,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp137,  (int32_t)3, tmp138);
	Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1000,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp138, tmp51, tmp139,  consSF);
	MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000, tmp139, tmp52, tmp140);
	Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000, tmp140, tmp141);
	AvgPool( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000,  (int32_t)13,  (int32_t)13,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000, tmp141, tmp142);
	for(int i=0;i<1000;i++){
		cout<<tmp142[0][0][0][i]<<" ";
	}
	cout<<endl;
}
return 0;
}

