#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include<fstream>

using namespace std;

uint32_t public_lrshift(uint32_t x, uint32_t y){
return (x >> y);
}

int32_t public_lrshift(int32_t x, uint32_t y){
return ((int32_t)(((uint32_t)x) >> y));
}

uint64_t public_lrshift(uint64_t x, uint64_t y){
return (x >> y);
}

int64_t public_lrshift(int64_t x, uint64_t y){
return ((int64_t)(((uint64_t)x) >> y));
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


uint32_t signedgtbl(int64_t x, int64_t y){

uint64_t ux = x;

uint64_t uy = y;

uint64_t signBitX = (x & ( (int64_t)1 <<  (int64_t)63));

uint64_t signBitY = (y & ( (int64_t)1 <<  (int64_t)63));
return ((signBitX ^ signBitY) >  (uint64_t)0) ? (signBitX >  (uint64_t)0) ? 0 : 1 : (ux > uy);
}

int64_t signedarshiftbl(int64_t x, uint64_t y){

uint64_t ux = x;

uint64_t signBitX = (x & ( (int64_t)1 <<  (int64_t)63));
return (signBitX >  (uint64_t)0) ? ( (uint64_t)0 - (( (uint64_t)0 - ux) >> y)) : (ux >> y);
}

uint32_t unsignedltbl(uint64_t x, uint64_t y){
return (y > x);
}

uint32_t signedltbl(int64_t x, int64_t y){
return (y > x);
}

uint32_t unsignedleqbl(uint64_t x, uint64_t y){
return ! (x > y);
}

uint32_t signedleqbl(int64_t x, int64_t y){
return ! (x > y);
}

uint32_t unsignedgeqbl(uint64_t x, uint64_t y){
return ! (y > x);
}

uint32_t signedgeqbl(int64_t x, int64_t y){
return ! (y > x);
}

uint32_t unsignedequalsbl(uint64_t x, uint64_t y){
return (! (x < y) && ! (y < x));
}

uint32_t signedequalsbl(int64_t x, int64_t y){
return (! (x < y) && ! (y < x));
}

uint64_t longDivision(uint64_t x, uint64_t y, uint32_t getQuotient){

uint64_t q =  (uint64_t)0;

uint64_t divisor =  (uint64_t)0;

uint32_t cond = 0;
for (uint32_t iter =  (int32_t)0; iter <  (int32_t)64; iter++){

uint64_t i = ( (int64_t)63 - iter);
divisor = (divisor <<  (uint64_t)1);
divisor = (divisor + (public_lrshift((x & ( (uint64_t)1 << i)), i)));
cond = (divisor >= y);
divisor = cond ? (divisor - y) : divisor;
q = (q <<  (uint64_t)1);
q = cond ? (q +  (uint64_t)1) : q;
}
return getQuotient ? q : divisor;
}

uint64_t unsigneddivbl(uint64_t x, uint64_t y){
return longDivision(x, y, 1);
}

uint64_t unsigneddival(uint64_t x, uint64_t y){

uint64_t bx = x;

uint64_t by = y;
return (bx / by);
}

int64_t signeddivbl(int64_t x, int64_t y){

uint32_t isXNeg = (x <  (int64_t)0);

uint32_t isYNeg = (y <  (int64_t)0);

uint64_t ux = isXNeg ? ( (int64_t)0 - x) : x;

uint64_t uy = isYNeg ? ( (int64_t)0 - y) : y;

uint64_t ures = (ux / uy);

uint32_t isResNeg = (isXNeg ^ isYNeg);
return isResNeg ? ( (uint64_t)0 - ures) : ures;
}

int64_t signeddival(int64_t x, int64_t y){

int64_t bx = x;

int64_t by = y;
return (bx / by);
}

uint64_t unsignedmodbl(uint64_t x, uint64_t y){
return longDivision(x, y, 0);
}

uint64_t unsignedmodal(uint64_t x, uint64_t y){

uint64_t bx = x;

uint64_t by = y;
return (bx % by);
}

int64_t signedmodbl(int64_t x, int64_t y){

uint32_t isXNeg = (x <  (int64_t)0);

uint32_t isYNeg = (y <  (int64_t)0);

uint64_t ux = isXNeg ? ( (int64_t)0 - x) : x;

uint64_t uy = isYNeg ? ( (int64_t)0 - y) : y;

uint64_t urem = (ux % uy);
return isXNeg ? ( (uint64_t)0 - urem) : urem;
}

int64_t signedmodal(int64_t x, int64_t y){

int64_t bx = x;

int64_t by = y;
return (bx % by);
}

void MatMulCSF2D(int32_t i, int32_t j, int32_t k, auto& A, auto& B, auto& C, int64_t consSF){
cout<<"Matmul"<<endl;
for (uint32_t i1 =  (int32_t)0; i1 < i; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < k; i2++){
C[i1][i2] =  (int64_t)0;
for (uint32_t i3 =  (int32_t)0; i3 < j; i3++){
C[i1][i2] = (C[i1][i2] + (A[i1][i3] * B[i3][i2]));
}
cout<<C[i1][i2]<<" ";
C[i1][i2] = (C[i1][i2] >> consSF);
}
}
cout<<endl;
}

void ArgMax1(int32_t outArrS1, int32_t inArrS1, int32_t inArrS2, auto& inArr, int32_t dim, auto& outArr){
for (uint32_t od =  (int32_t)0; od < inArrS1; od++){

int64_t maxi = inArr[od][ (int32_t)0];

int64_t maxiIdx =  (int64_t)0;
for (uint32_t i =  (int32_t)0; i < inArrS2; i++){

int64_t iL = i;
maxiIdx = (inArr[od][i] > maxi) ? iL : maxiIdx;
maxi = (inArr[od][i] > maxi) ? inArr[od][i] : maxi;
}
outArr[od] = maxiIdx;
}
}

void Relu2(int32_t s1, int32_t s2, auto& inArr, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = (inArr[i1][i2] >  (int64_t)0) ? inArr[i1][i2] :  (int64_t)0;
}
}
}

void Relu4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& inArr, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
outArr[i1][i2][i3][i4] = (inArr[i1][i2][i3][i4] >  (int64_t)0) ? inArr[i1][i2][i3][i4] :  (int64_t)0;
}
}
}
}
}

void ElemWiseMul2(int32_t s1, int32_t s2, auto& arr1, auto& arr2, auto& outArr, int64_t shrout){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = ((arr1[i1][i2] * arr2[i1][i2]) >> shrout);
}
}
}

void ElemWiseMul4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& arr1, auto& arr2, auto& outArr, int64_t shrout){
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

void ElemWiseDiv2(int32_t s1, int32_t s2, auto& arr1, auto& arr2, auto& outArr, int64_t shrout){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = ((arr1[i1][i2] / arr2[i1][i2]) << shrout);
}
}
}

void Floor2(int32_t s1, int32_t s2, auto& inArr, auto& outArr, int64_t curSF){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){

int64_t mask = ~ (( (int64_t)1 << curSF) -  (int64_t)1);
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

int64_t maxi =  (int64_t)0;
if ((((leftTopCornerH <  (int32_t)0) || (leftTopCornerH >= imgH)) || ((leftTopCornerW <  (int32_t)0) || (leftTopCornerW >= imgW)))) {
maxi =  (int64_t)0;
} else {
maxi = inArr[n][leftTopCornerH][leftTopCornerW][c];
}
for (uint32_t fh =  (int32_t)0; fh < ksizeH; fh++){
for (uint32_t fw =  (int32_t)0; fw < ksizeW; fw++){

int32_t curPosH = (leftTopCornerH + fh);

int32_t curPosW = (leftTopCornerW + fw);

int64_t temp =  (int64_t)0;
if ((((curPosH <  (int32_t)0) || (curPosH >= imgH)) || ((curPosW <  (int32_t)0) || (curPosW >= imgW)))) {
temp =  (int64_t)0;
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

auto filterAvg = make_vector<int64_t>(rows);

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

int64_t curFilterSum =  (int64_t)0;
for (uint32_t fh =  (int32_t)0; fh < ksizeH; fh++){
for (uint32_t fw =  (int32_t)0; fw < ksizeW; fw++){

int32_t curPosH = (leftTopCornerH + fh);

int32_t curPosW = (leftTopCornerW + fw);

int64_t temp =  (int64_t)0;
if ((((curPosH <  (int32_t)0) || (curPosH >= imgH)) || ((curPosW <  (int32_t)0) || (curPosW >= imgW)))) {
temp =  (int64_t)0;
} else {
temp = inArr[n][curPosH][curPosW][c];
}
curFilterSum = (curFilterSum + temp);
}
}

int64_t ksizeH64 = ksizeH;

int64_t ksizeW64 = ksizeW;

int64_t filterSz64 = (ksizeH64 * ksizeW64);

int64_t curFilterAvg = (curFilterSum / filterSz64);
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

void TempFusedBatchNorm4411(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& inArr, int32_t vecS1, auto& multArr, auto& biasArr, auto& outputArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){

int64_t t1 = (inArr[i1][i2][i3][i4] * multArr[i4]);

int64_t t2 = (t1 >>  (int64_t)15);
outputArr[i1][i2][i3][i4] = (t2 + biasArr[i4]);
}
}
}
}
}

void ScalarMul4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int64_t scalar, auto& inputArr, auto& outputArr, int64_t consSF){
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

int64_t summ =  (int64_t)0;
for (uint32_t i =  (int32_t)0; i < inS2; i++){
for (uint32_t j =  (int32_t)0; j < inS3; j++){
summ = (summ + inputArr[i1][i][j][i2]);
}
}

int64_t numElem = (inS2 * inS3);
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

void CreateTensor1(int32_t s1, int64_t val, auto& arr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
arr[i1] = val;
}
}

void CreateTensor2(int32_t s1, int32_t s2, int64_t val, auto& arr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
arr[i1][i2] = val;
}
}
}

void CreateTensor4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int64_t val, auto& arr){
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

void RandomUniform2(int32_t s1, int32_t s2, int64_t dataType, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] =  (int64_t)100;
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

int64_t val =  (int64_t)0;
for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
if ((((curPosH <  (int32_t)0) || (curPosH >= H)) || ((curPosW <  (int32_t)0) || (curPosW >= W)))) {
val =  (int64_t)0;
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

void Conv2DCSF(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, auto& inputArr, auto& filterArr, auto& outArr, int64_t consSF){

int32_t reshapedFilterRows = CO;

int32_t reshapedFilterCols = ((FH * FW) * CI);

int32_t reshapedIPRows = ((FH * FW) * CI);

int32_t newH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) +  (int32_t)1);

int32_t newW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) +  (int32_t)1);

int32_t reshapedIPCols = ((N * newH) * newW);

auto filterReshaped = make_vector<int64_t>(reshapedFilterRows, reshapedFilterCols);

auto inputReshaped = make_vector<int64_t>(reshapedIPRows, reshapedIPCols);

auto matmulOP = make_vector<int64_t>(reshapedFilterRows, reshapedIPCols);
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
outArr[i][j][k][l] =  (int64_t)0;
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


int main () {

auto tmp607 = make_vector<int64_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64);

auto tmp608 = make_vector<int64_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64);

auto tmp609 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp610 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp611 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp612 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp613 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp614 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp615 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp616 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32);

auto tmp617 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96);

auto tmp618 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96);

auto tmp619 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96);

auto tmp620 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp621 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp622 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp623 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32);

auto tmp624 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp625 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp626 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp627 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp628 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp629 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp630 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32);

auto tmp631 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160);

auto tmp632 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160);

auto tmp633 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160);

auto tmp634 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp635 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp636 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp637 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32);

auto tmp638 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192);

auto tmp639 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192);

auto tmp640 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192);

auto tmp641 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp642 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp643 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp644 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32);

auto tmp645 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224);

auto tmp646 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224);

auto tmp647 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224);

auto tmp648 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp649 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp650 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp651 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32);

auto tmp652 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp653 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp654 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp655 = make_vector<int64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp656 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp657 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp658 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp659 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp660 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp661 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp662 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp663 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160);

auto tmp664 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160);

auto tmp665 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160);

auto tmp666 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp667 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp668 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp669 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp670 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192);

auto tmp671 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192);

auto tmp672 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192);

auto tmp673 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp674 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp675 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp676 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp677 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224);

auto tmp678 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224);

auto tmp679 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224);

auto tmp680 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp681 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp682 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp683 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp684 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);

auto tmp685 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);

auto tmp686 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);

auto tmp687 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp688 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp689 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp690 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp691 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288);

auto tmp692 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288);

auto tmp693 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288);

auto tmp694 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp695 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp696 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp697 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp698 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320);

auto tmp699 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320);

auto tmp700 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320);

auto tmp701 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp702 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp703 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp704 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp705 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352);

auto tmp706 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352);

auto tmp707 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352);

auto tmp708 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp709 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp710 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp711 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp712 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384);

auto tmp713 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384);

auto tmp714 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384);

auto tmp715 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp716 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp717 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp718 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp719 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416);

auto tmp720 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416);

auto tmp721 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416);

auto tmp722 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp723 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp724 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp725 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp726 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448);

auto tmp727 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448);

auto tmp728 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448);

auto tmp729 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp730 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp731 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp732 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp733 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480);

auto tmp734 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480);

auto tmp735 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480);

auto tmp736 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp737 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp738 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp739 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32);

auto tmp740 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp741 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp742 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp743 = make_vector<int64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);

auto tmp744 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp745 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp746 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp747 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp748 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp749 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp750 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp751 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288);

auto tmp752 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288);

auto tmp753 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288);

auto tmp754 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp755 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp756 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp757 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp758 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320);

auto tmp759 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320);

auto tmp760 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320);

auto tmp761 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp762 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp763 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp764 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp765 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352);

auto tmp766 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352);

auto tmp767 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352);

auto tmp768 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp769 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp770 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp771 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp772 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384);

auto tmp773 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384);

auto tmp774 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384);

auto tmp775 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp776 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp777 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp778 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp779 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416);

auto tmp780 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416);

auto tmp781 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416);

auto tmp782 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp783 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp784 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp785 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp786 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448);

auto tmp787 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448);

auto tmp788 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448);

auto tmp789 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp790 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp791 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp792 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp793 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480);

auto tmp794 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480);

auto tmp795 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480);

auto tmp796 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp797 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp798 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp799 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp800 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);

auto tmp801 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);

auto tmp802 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);

auto tmp803 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp804 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp805 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp806 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp807 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544);

auto tmp808 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544);

auto tmp809 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544);

auto tmp810 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp811 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp812 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp813 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp814 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576);

auto tmp815 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576);

auto tmp816 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576);

auto tmp817 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp818 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp819 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp820 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp821 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608);

auto tmp822 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608);

auto tmp823 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608);

auto tmp824 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp825 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp826 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp827 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp828 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640);

auto tmp829 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640);

auto tmp830 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640);

auto tmp831 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp832 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp833 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp834 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp835 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672);

auto tmp836 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672);

auto tmp837 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672);

auto tmp838 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp839 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp840 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp841 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp842 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704);

auto tmp843 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704);

auto tmp844 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704);

auto tmp845 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp846 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp847 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp848 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp849 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736);

auto tmp850 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736);

auto tmp851 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736);

auto tmp852 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp853 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp854 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp855 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp856 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768);

auto tmp857 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768);

auto tmp858 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768);

auto tmp859 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp860 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp861 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp862 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp863 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800);

auto tmp864 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800);

auto tmp865 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800);

auto tmp866 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp867 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp868 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp869 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp870 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832);

auto tmp871 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832);

auto tmp872 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832);

auto tmp873 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp874 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp875 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp876 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp877 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864);

auto tmp878 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864);

auto tmp879 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864);

auto tmp880 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp881 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp882 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp883 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp884 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896);

auto tmp885 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896);

auto tmp886 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896);

auto tmp887 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp888 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp889 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp890 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp891 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928);

auto tmp892 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928);

auto tmp893 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928);

auto tmp894 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp895 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp896 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp897 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp898 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960);

auto tmp899 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960);

auto tmp900 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960);

auto tmp901 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp902 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp903 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp904 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp905 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992);

auto tmp906 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992);

auto tmp907 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992);

auto tmp908 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp909 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp910 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128);

auto tmp911 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32);

auto tmp912 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp913 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp914 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp915 = make_vector<int64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);

auto tmp916 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp917 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp918 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp919 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp920 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp921 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp922 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp923 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544);

auto tmp924 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544);

auto tmp925 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544);

auto tmp926 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp927 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp928 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp929 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp930 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576);

auto tmp931 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576);

auto tmp932 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576);

auto tmp933 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp934 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp935 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp936 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp937 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608);

auto tmp938 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608);

auto tmp939 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608);

auto tmp940 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp941 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp942 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp943 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp944 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640);

auto tmp945 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640);

auto tmp946 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640);

auto tmp947 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp948 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp949 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp950 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp951 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672);

auto tmp952 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672);

auto tmp953 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672);

auto tmp954 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp955 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp956 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp957 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp958 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704);

auto tmp959 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704);

auto tmp960 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704);

auto tmp961 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp962 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp963 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp964 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp965 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736);

auto tmp966 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736);

auto tmp967 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736);

auto tmp968 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp969 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp970 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp971 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp972 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768);

auto tmp973 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768);

auto tmp974 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768);

auto tmp975 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp976 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp977 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp978 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp979 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800);

auto tmp980 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800);

auto tmp981 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800);

auto tmp982 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp983 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp984 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp985 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp986 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832);

auto tmp987 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832);

auto tmp988 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832);

auto tmp989 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp990 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp991 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp992 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp993 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864);

auto tmp994 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864);

auto tmp995 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864);

auto tmp996 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp997 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp998 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp999 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp1000 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896);

auto tmp1001 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896);

auto tmp1002 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896);

auto tmp1003 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1004 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1005 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1006 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp1007 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928);

auto tmp1008 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928);

auto tmp1009 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928);

auto tmp1010 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1011 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1012 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1013 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp1014 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960);

auto tmp1015 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960);

auto tmp1016 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960);

auto tmp1017 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1018 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1019 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1020 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp1021 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992);

auto tmp1022 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992);

auto tmp1023 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992);

auto tmp1024 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1025 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1026 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128);

auto tmp1027 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32);

auto tmp1028 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024);

auto tmp1029 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024);

auto tmp1030 = make_vector<int64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024);

auto tmp1031 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1024);

auto tmp1032 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000);

auto tmp1033 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000);

auto tmp0 = make_vector<int64_t>( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3);
/* Variable to read the clear value corresponding to the input variable tmp0 at (1103,1-1103,47) */
uint64_t __tmp_in_tmp0;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)224; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)224; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)3; i3++){
cin >> __tmp_in_tmp0;
tmp0[i0][i1][i2][i3] = __tmp_in_tmp0;
}
}
}
}

auto tmp1 = make_vector<int64_t>( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp1 at (1105,1-1105,44) */
uint64_t __tmp_in_tmp1;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)7; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)7; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)3; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
cin >> __tmp_in_tmp1;
tmp1[i0][i1][i2][i3] = __tmp_in_tmp1;
}
}
}
}

auto tmp2 = make_vector<int64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp2 at (1107,1-1107,35) */
uint64_t __tmp_in_tmp2;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp2;
tmp2[i0] = __tmp_in_tmp2;
}

auto tmp3 = make_vector<int64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp3 at (1109,1-1109,35) */
uint64_t __tmp_in_tmp3;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp3;
tmp3[i0] = __tmp_in_tmp3;
}

auto tmp4 = make_vector<int64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp4 at (1111,1-1111,35) */
uint64_t __tmp_in_tmp4;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp4;
tmp4[i0] = __tmp_in_tmp4;
}

auto tmp5 = make_vector<int64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp5 at (1113,1-1113,35) */
uint64_t __tmp_in_tmp5;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp5;
tmp5[i0] = __tmp_in_tmp5;
}

auto tmp6 = make_vector<int64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp6 at (1115,1-1115,35) */
uint64_t __tmp_in_tmp6;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp6;
tmp6[i0] = __tmp_in_tmp6;
}

auto tmp7 = make_vector<int64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp7 at (1117,1-1117,35) */
uint64_t __tmp_in_tmp7;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp7;
tmp7[i0] = __tmp_in_tmp7;
}

auto tmp8 = make_vector<int64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp8 at (1119,1-1119,35) */
uint64_t __tmp_in_tmp8;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp8;
tmp8[i0] = __tmp_in_tmp8;
}

auto tmp9 = make_vector<int64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp9 at (1121,1-1121,35) */
uint64_t __tmp_in_tmp9;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
cin >> __tmp_in_tmp9;
tmp9[i0] = __tmp_in_tmp9;
}

auto tmp10 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp10 at (1123,1-1123,47) */
uint64_t __tmp_in_tmp10;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp10;
tmp10[i0][i1][i2][i3] = __tmp_in_tmp10;
}
}
}
}

auto tmp11 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp11 at (1125,1-1125,37) */
uint64_t __tmp_in_tmp11;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp11;
tmp11[i0] = __tmp_in_tmp11;
}

auto tmp12 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp12 at (1127,1-1127,37) */
uint64_t __tmp_in_tmp12;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp12;
tmp12[i0] = __tmp_in_tmp12;
}

auto tmp13 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp13 at (1129,1-1129,37) */
uint64_t __tmp_in_tmp13;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp13;
tmp13[i0] = __tmp_in_tmp13;
}

auto tmp14 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp14 at (1131,1-1131,37) */
uint64_t __tmp_in_tmp14;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp14;
tmp14[i0] = __tmp_in_tmp14;
}

auto tmp15 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp15 at (1133,1-1133,47) */
uint64_t __tmp_in_tmp15;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp15;
tmp15[i0][i1][i2][i3] = __tmp_in_tmp15;
}
}
}
}

auto tmp16 = make_vector<int64_t>( (int32_t)96);
/* Variable to read the clear value corresponding to the input variable tmp16 at (1135,1-1135,36) */
uint64_t __tmp_in_tmp16;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)96; i0++){
cin >> __tmp_in_tmp16;
tmp16[i0] = __tmp_in_tmp16;
}

auto tmp17 = make_vector<int64_t>( (int32_t)96);
/* Variable to read the clear value corresponding to the input variable tmp17 at (1137,1-1137,36) */
uint64_t __tmp_in_tmp17;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)96; i0++){
cin >> __tmp_in_tmp17;
tmp17[i0] = __tmp_in_tmp17;
}

auto tmp18 = make_vector<int64_t>( (int32_t)96);
/* Variable to read the clear value corresponding to the input variable tmp18 at (1139,1-1139,36) */
uint64_t __tmp_in_tmp18;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)96; i0++){
cin >> __tmp_in_tmp18;
tmp18[i0] = __tmp_in_tmp18;
}

auto tmp19 = make_vector<int64_t>( (int32_t)96);
/* Variable to read the clear value corresponding to the input variable tmp19 at (1141,1-1141,36) */
uint64_t __tmp_in_tmp19;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)96; i0++){
cin >> __tmp_in_tmp19;
tmp19[i0] = __tmp_in_tmp19;
}

auto tmp20 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)96,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp20 at (1143,1-1143,47) */
uint64_t __tmp_in_tmp20;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)96; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp20;
tmp20[i0][i1][i2][i3] = __tmp_in_tmp20;
}
}
}
}

auto tmp21 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp21 at (1145,1-1145,37) */
uint64_t __tmp_in_tmp21;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp21;
tmp21[i0] = __tmp_in_tmp21;
}

auto tmp22 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp22 at (1147,1-1147,37) */
uint64_t __tmp_in_tmp22;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp22;
tmp22[i0] = __tmp_in_tmp22;
}

auto tmp23 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp23 at (1149,1-1149,37) */
uint64_t __tmp_in_tmp23;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp23;
tmp23[i0] = __tmp_in_tmp23;
}

auto tmp24 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp24 at (1151,1-1151,37) */
uint64_t __tmp_in_tmp24;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp24;
tmp24[i0] = __tmp_in_tmp24;
}

auto tmp25 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp25 at (1153,1-1153,47) */
uint64_t __tmp_in_tmp25;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp25;
tmp25[i0][i1][i2][i3] = __tmp_in_tmp25;
}
}
}
}

auto tmp26 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp26 at (1155,1-1155,37) */
uint64_t __tmp_in_tmp26;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp26;
tmp26[i0] = __tmp_in_tmp26;
}

auto tmp27 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp27 at (1157,1-1157,37) */
uint64_t __tmp_in_tmp27;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp27;
tmp27[i0] = __tmp_in_tmp27;
}

auto tmp28 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp28 at (1159,1-1159,37) */
uint64_t __tmp_in_tmp28;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp28;
tmp28[i0] = __tmp_in_tmp28;
}

auto tmp29 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp29 at (1161,1-1161,37) */
uint64_t __tmp_in_tmp29;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp29;
tmp29[i0] = __tmp_in_tmp29;
}

auto tmp30 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp30 at (1163,1-1163,48) */
uint64_t __tmp_in_tmp30;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp30;
tmp30[i0][i1][i2][i3] = __tmp_in_tmp30;
}
}
}
}

auto tmp31 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp31 at (1165,1-1165,37) */
uint64_t __tmp_in_tmp31;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp31;
tmp31[i0] = __tmp_in_tmp31;
}

auto tmp32 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp32 at (1167,1-1167,37) */
uint64_t __tmp_in_tmp32;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp32;
tmp32[i0] = __tmp_in_tmp32;
}

auto tmp33 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp33 at (1169,1-1169,37) */
uint64_t __tmp_in_tmp33;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp33;
tmp33[i0] = __tmp_in_tmp33;
}

auto tmp34 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp34 at (1171,1-1171,37) */
uint64_t __tmp_in_tmp34;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp34;
tmp34[i0] = __tmp_in_tmp34;
}

auto tmp35 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp35 at (1173,1-1173,47) */
uint64_t __tmp_in_tmp35;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp35;
tmp35[i0][i1][i2][i3] = __tmp_in_tmp35;
}
}
}
}

auto tmp36 = make_vector<int64_t>( (int32_t)160);
/* Variable to read the clear value corresponding to the input variable tmp36 at (1175,1-1175,37) */
uint64_t __tmp_in_tmp36;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
cin >> __tmp_in_tmp36;
tmp36[i0] = __tmp_in_tmp36;
}

auto tmp37 = make_vector<int64_t>( (int32_t)160);
/* Variable to read the clear value corresponding to the input variable tmp37 at (1177,1-1177,37) */
uint64_t __tmp_in_tmp37;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
cin >> __tmp_in_tmp37;
tmp37[i0] = __tmp_in_tmp37;
}

auto tmp38 = make_vector<int64_t>( (int32_t)160);
/* Variable to read the clear value corresponding to the input variable tmp38 at (1179,1-1179,37) */
uint64_t __tmp_in_tmp38;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
cin >> __tmp_in_tmp38;
tmp38[i0] = __tmp_in_tmp38;
}

auto tmp39 = make_vector<int64_t>( (int32_t)160);
/* Variable to read the clear value corresponding to the input variable tmp39 at (1181,1-1181,37) */
uint64_t __tmp_in_tmp39;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
cin >> __tmp_in_tmp39;
tmp39[i0] = __tmp_in_tmp39;
}

auto tmp40 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)160,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp40 at (1183,1-1183,48) */
uint64_t __tmp_in_tmp40;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)160; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp40;
tmp40[i0][i1][i2][i3] = __tmp_in_tmp40;
}
}
}
}

auto tmp41 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp41 at (1185,1-1185,37) */
uint64_t __tmp_in_tmp41;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp41;
tmp41[i0] = __tmp_in_tmp41;
}

auto tmp42 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp42 at (1187,1-1187,37) */
uint64_t __tmp_in_tmp42;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp42;
tmp42[i0] = __tmp_in_tmp42;
}

auto tmp43 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp43 at (1189,1-1189,37) */
uint64_t __tmp_in_tmp43;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp43;
tmp43[i0] = __tmp_in_tmp43;
}

auto tmp44 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp44 at (1191,1-1191,37) */
uint64_t __tmp_in_tmp44;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp44;
tmp44[i0] = __tmp_in_tmp44;
}

auto tmp45 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp45 at (1193,1-1193,47) */
uint64_t __tmp_in_tmp45;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp45;
tmp45[i0][i1][i2][i3] = __tmp_in_tmp45;
}
}
}
}

auto tmp46 = make_vector<int64_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp46 at (1195,1-1195,37) */
uint64_t __tmp_in_tmp46;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp46;
tmp46[i0] = __tmp_in_tmp46;
}

auto tmp47 = make_vector<int64_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp47 at (1197,1-1197,37) */
uint64_t __tmp_in_tmp47;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp47;
tmp47[i0] = __tmp_in_tmp47;
}

auto tmp48 = make_vector<int64_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp48 at (1199,1-1199,37) */
uint64_t __tmp_in_tmp48;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp48;
tmp48[i0] = __tmp_in_tmp48;
}

auto tmp49 = make_vector<int64_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp49 at (1201,1-1201,37) */
uint64_t __tmp_in_tmp49;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp49;
tmp49[i0] = __tmp_in_tmp49;
}

auto tmp50 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)192,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp50 at (1203,1-1203,48) */
uint64_t __tmp_in_tmp50;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)192; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp50;
tmp50[i0][i1][i2][i3] = __tmp_in_tmp50;
}
}
}
}

auto tmp51 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp51 at (1205,1-1205,37) */
uint64_t __tmp_in_tmp51;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp51;
tmp51[i0] = __tmp_in_tmp51;
}

auto tmp52 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp52 at (1207,1-1207,37) */
uint64_t __tmp_in_tmp52;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp52;
tmp52[i0] = __tmp_in_tmp52;
}

auto tmp53 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp53 at (1209,1-1209,37) */
uint64_t __tmp_in_tmp53;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp53;
tmp53[i0] = __tmp_in_tmp53;
}

auto tmp54 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp54 at (1211,1-1211,37) */
uint64_t __tmp_in_tmp54;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp54;
tmp54[i0] = __tmp_in_tmp54;
}

auto tmp55 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp55 at (1213,1-1213,47) */
uint64_t __tmp_in_tmp55;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp55;
tmp55[i0][i1][i2][i3] = __tmp_in_tmp55;
}
}
}
}

auto tmp56 = make_vector<int64_t>( (int32_t)224);
/* Variable to read the clear value corresponding to the input variable tmp56 at (1215,1-1215,37) */
uint64_t __tmp_in_tmp56;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
cin >> __tmp_in_tmp56;
tmp56[i0] = __tmp_in_tmp56;
}

auto tmp57 = make_vector<int64_t>( (int32_t)224);
/* Variable to read the clear value corresponding to the input variable tmp57 at (1217,1-1217,37) */
uint64_t __tmp_in_tmp57;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
cin >> __tmp_in_tmp57;
tmp57[i0] = __tmp_in_tmp57;
}

auto tmp58 = make_vector<int64_t>( (int32_t)224);
/* Variable to read the clear value corresponding to the input variable tmp58 at (1219,1-1219,37) */
uint64_t __tmp_in_tmp58;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
cin >> __tmp_in_tmp58;
tmp58[i0] = __tmp_in_tmp58;
}

auto tmp59 = make_vector<int64_t>( (int32_t)224);
/* Variable to read the clear value corresponding to the input variable tmp59 at (1221,1-1221,37) */
uint64_t __tmp_in_tmp59;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
cin >> __tmp_in_tmp59;
tmp59[i0] = __tmp_in_tmp59;
}

auto tmp60 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)224,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp60 at (1223,1-1223,48) */
uint64_t __tmp_in_tmp60;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)224; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp60;
tmp60[i0][i1][i2][i3] = __tmp_in_tmp60;
}
}
}
}

auto tmp61 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp61 at (1225,1-1225,37) */
uint64_t __tmp_in_tmp61;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp61;
tmp61[i0] = __tmp_in_tmp61;
}

auto tmp62 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp62 at (1227,1-1227,37) */
uint64_t __tmp_in_tmp62;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp62;
tmp62[i0] = __tmp_in_tmp62;
}

auto tmp63 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp63 at (1229,1-1229,37) */
uint64_t __tmp_in_tmp63;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp63;
tmp63[i0] = __tmp_in_tmp63;
}

auto tmp64 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp64 at (1231,1-1231,37) */
uint64_t __tmp_in_tmp64;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp64;
tmp64[i0] = __tmp_in_tmp64;
}

auto tmp65 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp65 at (1233,1-1233,47) */
uint64_t __tmp_in_tmp65;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp65;
tmp65[i0][i1][i2][i3] = __tmp_in_tmp65;
}
}
}
}

auto tmp66 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp66 at (1235,1-1235,37) */
uint64_t __tmp_in_tmp66;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp66;
tmp66[i0] = __tmp_in_tmp66;
}

auto tmp67 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp67 at (1237,1-1237,37) */
uint64_t __tmp_in_tmp67;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp67;
tmp67[i0] = __tmp_in_tmp67;
}

auto tmp68 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp68 at (1239,1-1239,37) */
uint64_t __tmp_in_tmp68;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp68;
tmp68[i0] = __tmp_in_tmp68;
}

auto tmp69 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp69 at (1241,1-1241,37) */
uint64_t __tmp_in_tmp69;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp69;
tmp69[i0] = __tmp_in_tmp69;
}

auto tmp70 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp70 at (1243,1-1243,48) */
uint64_t __tmp_in_tmp70;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp70;
tmp70[i0][i1][i2][i3] = __tmp_in_tmp70;
}
}
}
}

auto tmp71 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp71 at (1245,1-1245,37) */
uint64_t __tmp_in_tmp71;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp71;
tmp71[i0] = __tmp_in_tmp71;
}

auto tmp72 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp72 at (1247,1-1247,37) */
uint64_t __tmp_in_tmp72;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp72;
tmp72[i0] = __tmp_in_tmp72;
}

auto tmp73 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp73 at (1249,1-1249,37) */
uint64_t __tmp_in_tmp73;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp73;
tmp73[i0] = __tmp_in_tmp73;
}

auto tmp74 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp74 at (1251,1-1251,37) */
uint64_t __tmp_in_tmp74;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp74;
tmp74[i0] = __tmp_in_tmp74;
}

auto tmp75 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp75 at (1253,1-1253,48) */
uint64_t __tmp_in_tmp75;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp75;
tmp75[i0][i1][i2][i3] = __tmp_in_tmp75;
}
}
}
}

auto tmp76 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp76 at (1255,1-1255,37) */
uint64_t __tmp_in_tmp76;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp76;
tmp76[i0] = __tmp_in_tmp76;
}

auto tmp77 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp77 at (1257,1-1257,37) */
uint64_t __tmp_in_tmp77;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp77;
tmp77[i0] = __tmp_in_tmp77;
}

auto tmp78 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp78 at (1259,1-1259,37) */
uint64_t __tmp_in_tmp78;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp78;
tmp78[i0] = __tmp_in_tmp78;
}

auto tmp79 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp79 at (1261,1-1261,37) */
uint64_t __tmp_in_tmp79;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp79;
tmp79[i0] = __tmp_in_tmp79;
}

auto tmp80 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp80 at (1263,1-1263,47) */
uint64_t __tmp_in_tmp80;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp80;
tmp80[i0][i1][i2][i3] = __tmp_in_tmp80;
}
}
}
}

auto tmp81 = make_vector<int64_t>( (int32_t)160);
/* Variable to read the clear value corresponding to the input variable tmp81 at (1265,1-1265,37) */
uint64_t __tmp_in_tmp81;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
cin >> __tmp_in_tmp81;
tmp81[i0] = __tmp_in_tmp81;
}

auto tmp82 = make_vector<int64_t>( (int32_t)160);
/* Variable to read the clear value corresponding to the input variable tmp82 at (1267,1-1267,37) */
uint64_t __tmp_in_tmp82;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
cin >> __tmp_in_tmp82;
tmp82[i0] = __tmp_in_tmp82;
}

auto tmp83 = make_vector<int64_t>( (int32_t)160);
/* Variable to read the clear value corresponding to the input variable tmp83 at (1269,1-1269,37) */
uint64_t __tmp_in_tmp83;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
cin >> __tmp_in_tmp83;
tmp83[i0] = __tmp_in_tmp83;
}

auto tmp84 = make_vector<int64_t>( (int32_t)160);
/* Variable to read the clear value corresponding to the input variable tmp84 at (1271,1-1271,37) */
uint64_t __tmp_in_tmp84;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
cin >> __tmp_in_tmp84;
tmp84[i0] = __tmp_in_tmp84;
}

auto tmp85 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)160,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp85 at (1273,1-1273,48) */
uint64_t __tmp_in_tmp85;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)160; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp85;
tmp85[i0][i1][i2][i3] = __tmp_in_tmp85;
}
}
}
}

auto tmp86 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp86 at (1275,1-1275,37) */
uint64_t __tmp_in_tmp86;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp86;
tmp86[i0] = __tmp_in_tmp86;
}

auto tmp87 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp87 at (1277,1-1277,37) */
uint64_t __tmp_in_tmp87;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp87;
tmp87[i0] = __tmp_in_tmp87;
}

auto tmp88 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp88 at (1279,1-1279,37) */
uint64_t __tmp_in_tmp88;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp88;
tmp88[i0] = __tmp_in_tmp88;
}

auto tmp89 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp89 at (1281,1-1281,37) */
uint64_t __tmp_in_tmp89;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp89;
tmp89[i0] = __tmp_in_tmp89;
}

auto tmp90 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp90 at (1283,1-1283,47) */
uint64_t __tmp_in_tmp90;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp90;
tmp90[i0][i1][i2][i3] = __tmp_in_tmp90;
}
}
}
}

auto tmp91 = make_vector<int64_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp91 at (1285,1-1285,37) */
uint64_t __tmp_in_tmp91;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp91;
tmp91[i0] = __tmp_in_tmp91;
}

auto tmp92 = make_vector<int64_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp92 at (1287,1-1287,37) */
uint64_t __tmp_in_tmp92;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp92;
tmp92[i0] = __tmp_in_tmp92;
}

auto tmp93 = make_vector<int64_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp93 at (1289,1-1289,37) */
uint64_t __tmp_in_tmp93;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp93;
tmp93[i0] = __tmp_in_tmp93;
}

auto tmp94 = make_vector<int64_t>( (int32_t)192);
/* Variable to read the clear value corresponding to the input variable tmp94 at (1291,1-1291,37) */
uint64_t __tmp_in_tmp94;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
cin >> __tmp_in_tmp94;
tmp94[i0] = __tmp_in_tmp94;
}

auto tmp95 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)192,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp95 at (1293,1-1293,48) */
uint64_t __tmp_in_tmp95;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)192; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp95;
tmp95[i0][i1][i2][i3] = __tmp_in_tmp95;
}
}
}
}

auto tmp96 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp96 at (1295,1-1295,37) */
uint64_t __tmp_in_tmp96;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp96;
tmp96[i0] = __tmp_in_tmp96;
}

auto tmp97 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp97 at (1297,1-1297,37) */
uint64_t __tmp_in_tmp97;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp97;
tmp97[i0] = __tmp_in_tmp97;
}

auto tmp98 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp98 at (1299,1-1299,37) */
uint64_t __tmp_in_tmp98;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp98;
tmp98[i0] = __tmp_in_tmp98;
}

auto tmp99 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp99 at (1301,1-1301,37) */
uint64_t __tmp_in_tmp99;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp99;
tmp99[i0] = __tmp_in_tmp99;
}

auto tmp100 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp100 at (1303,1-1303,48) */
uint64_t __tmp_in_tmp100;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp100;
tmp100[i0][i1][i2][i3] = __tmp_in_tmp100;
}
}
}
}

auto tmp101 = make_vector<int64_t>( (int32_t)224);
/* Variable to read the clear value corresponding to the input variable tmp101 at (1305,1-1305,38) */
uint64_t __tmp_in_tmp101;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
cin >> __tmp_in_tmp101;
tmp101[i0] = __tmp_in_tmp101;
}

auto tmp102 = make_vector<int64_t>( (int32_t)224);
/* Variable to read the clear value corresponding to the input variable tmp102 at (1307,1-1307,38) */
uint64_t __tmp_in_tmp102;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
cin >> __tmp_in_tmp102;
tmp102[i0] = __tmp_in_tmp102;
}

auto tmp103 = make_vector<int64_t>( (int32_t)224);
/* Variable to read the clear value corresponding to the input variable tmp103 at (1309,1-1309,38) */
uint64_t __tmp_in_tmp103;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
cin >> __tmp_in_tmp103;
tmp103[i0] = __tmp_in_tmp103;
}

auto tmp104 = make_vector<int64_t>( (int32_t)224);
/* Variable to read the clear value corresponding to the input variable tmp104 at (1311,1-1311,38) */
uint64_t __tmp_in_tmp104;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
cin >> __tmp_in_tmp104;
tmp104[i0] = __tmp_in_tmp104;
}

auto tmp105 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)224,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp105 at (1313,1-1313,49) */
uint64_t __tmp_in_tmp105;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)224; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp105;
tmp105[i0][i1][i2][i3] = __tmp_in_tmp105;
}
}
}
}

auto tmp106 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp106 at (1315,1-1315,38) */
uint64_t __tmp_in_tmp106;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp106;
tmp106[i0] = __tmp_in_tmp106;
}

auto tmp107 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp107 at (1317,1-1317,38) */
uint64_t __tmp_in_tmp107;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp107;
tmp107[i0] = __tmp_in_tmp107;
}

auto tmp108 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp108 at (1319,1-1319,38) */
uint64_t __tmp_in_tmp108;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp108;
tmp108[i0] = __tmp_in_tmp108;
}

auto tmp109 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp109 at (1321,1-1321,38) */
uint64_t __tmp_in_tmp109;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp109;
tmp109[i0] = __tmp_in_tmp109;
}

auto tmp110 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp110 at (1323,1-1323,48) */
uint64_t __tmp_in_tmp110;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp110;
tmp110[i0][i1][i2][i3] = __tmp_in_tmp110;
}
}
}
}

auto tmp111 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp111 at (1325,1-1325,38) */
uint64_t __tmp_in_tmp111;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp111;
tmp111[i0] = __tmp_in_tmp111;
}

auto tmp112 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp112 at (1327,1-1327,38) */
uint64_t __tmp_in_tmp112;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp112;
tmp112[i0] = __tmp_in_tmp112;
}

auto tmp113 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp113 at (1329,1-1329,38) */
uint64_t __tmp_in_tmp113;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp113;
tmp113[i0] = __tmp_in_tmp113;
}

auto tmp114 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp114 at (1331,1-1331,38) */
uint64_t __tmp_in_tmp114;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp114;
tmp114[i0] = __tmp_in_tmp114;
}

auto tmp115 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp115 at (1333,1-1333,49) */
uint64_t __tmp_in_tmp115;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp115;
tmp115[i0][i1][i2][i3] = __tmp_in_tmp115;
}
}
}
}

auto tmp116 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp116 at (1335,1-1335,38) */
uint64_t __tmp_in_tmp116;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp116;
tmp116[i0] = __tmp_in_tmp116;
}

auto tmp117 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp117 at (1337,1-1337,38) */
uint64_t __tmp_in_tmp117;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp117;
tmp117[i0] = __tmp_in_tmp117;
}

auto tmp118 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp118 at (1339,1-1339,38) */
uint64_t __tmp_in_tmp118;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp118;
tmp118[i0] = __tmp_in_tmp118;
}

auto tmp119 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp119 at (1341,1-1341,38) */
uint64_t __tmp_in_tmp119;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp119;
tmp119[i0] = __tmp_in_tmp119;
}

auto tmp120 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp120 at (1343,1-1343,48) */
uint64_t __tmp_in_tmp120;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp120;
tmp120[i0][i1][i2][i3] = __tmp_in_tmp120;
}
}
}
}

auto tmp121 = make_vector<int64_t>( (int32_t)288);
/* Variable to read the clear value corresponding to the input variable tmp121 at (1345,1-1345,38) */
uint64_t __tmp_in_tmp121;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
cin >> __tmp_in_tmp121;
tmp121[i0] = __tmp_in_tmp121;
}

auto tmp122 = make_vector<int64_t>( (int32_t)288);
/* Variable to read the clear value corresponding to the input variable tmp122 at (1347,1-1347,38) */
uint64_t __tmp_in_tmp122;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
cin >> __tmp_in_tmp122;
tmp122[i0] = __tmp_in_tmp122;
}

auto tmp123 = make_vector<int64_t>( (int32_t)288);
/* Variable to read the clear value corresponding to the input variable tmp123 at (1349,1-1349,38) */
uint64_t __tmp_in_tmp123;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
cin >> __tmp_in_tmp123;
tmp123[i0] = __tmp_in_tmp123;
}

auto tmp124 = make_vector<int64_t>( (int32_t)288);
/* Variable to read the clear value corresponding to the input variable tmp124 at (1351,1-1351,38) */
uint64_t __tmp_in_tmp124;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
cin >> __tmp_in_tmp124;
tmp124[i0] = __tmp_in_tmp124;
}

auto tmp125 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)288,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp125 at (1353,1-1353,49) */
uint64_t __tmp_in_tmp125;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)288; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp125;
tmp125[i0][i1][i2][i3] = __tmp_in_tmp125;
}
}
}
}

auto tmp126 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp126 at (1355,1-1355,38) */
uint64_t __tmp_in_tmp126;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp126;
tmp126[i0] = __tmp_in_tmp126;
}

auto tmp127 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp127 at (1357,1-1357,38) */
uint64_t __tmp_in_tmp127;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp127;
tmp127[i0] = __tmp_in_tmp127;
}

auto tmp128 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp128 at (1359,1-1359,38) */
uint64_t __tmp_in_tmp128;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp128;
tmp128[i0] = __tmp_in_tmp128;
}

auto tmp129 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp129 at (1361,1-1361,38) */
uint64_t __tmp_in_tmp129;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp129;
tmp129[i0] = __tmp_in_tmp129;
}

auto tmp130 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp130 at (1363,1-1363,48) */
uint64_t __tmp_in_tmp130;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp130;
tmp130[i0][i1][i2][i3] = __tmp_in_tmp130;
}
}
}
}

auto tmp131 = make_vector<int64_t>( (int32_t)320);
/* Variable to read the clear value corresponding to the input variable tmp131 at (1365,1-1365,38) */
uint64_t __tmp_in_tmp131;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
cin >> __tmp_in_tmp131;
tmp131[i0] = __tmp_in_tmp131;
}

auto tmp132 = make_vector<int64_t>( (int32_t)320);
/* Variable to read the clear value corresponding to the input variable tmp132 at (1367,1-1367,38) */
uint64_t __tmp_in_tmp132;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
cin >> __tmp_in_tmp132;
tmp132[i0] = __tmp_in_tmp132;
}

auto tmp133 = make_vector<int64_t>( (int32_t)320);
/* Variable to read the clear value corresponding to the input variable tmp133 at (1369,1-1369,38) */
uint64_t __tmp_in_tmp133;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
cin >> __tmp_in_tmp133;
tmp133[i0] = __tmp_in_tmp133;
}

auto tmp134 = make_vector<int64_t>( (int32_t)320);
/* Variable to read the clear value corresponding to the input variable tmp134 at (1371,1-1371,38) */
uint64_t __tmp_in_tmp134;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
cin >> __tmp_in_tmp134;
tmp134[i0] = __tmp_in_tmp134;
}

auto tmp135 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)320,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp135 at (1373,1-1373,49) */
uint64_t __tmp_in_tmp135;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)320; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp135;
tmp135[i0][i1][i2][i3] = __tmp_in_tmp135;
}
}
}
}

auto tmp136 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp136 at (1375,1-1375,38) */
uint64_t __tmp_in_tmp136;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp136;
tmp136[i0] = __tmp_in_tmp136;
}

auto tmp137 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp137 at (1377,1-1377,38) */
uint64_t __tmp_in_tmp137;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp137;
tmp137[i0] = __tmp_in_tmp137;
}

auto tmp138 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp138 at (1379,1-1379,38) */
uint64_t __tmp_in_tmp138;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp138;
tmp138[i0] = __tmp_in_tmp138;
}

auto tmp139 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp139 at (1381,1-1381,38) */
uint64_t __tmp_in_tmp139;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp139;
tmp139[i0] = __tmp_in_tmp139;
}

auto tmp140 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp140 at (1383,1-1383,48) */
uint64_t __tmp_in_tmp140;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp140;
tmp140[i0][i1][i2][i3] = __tmp_in_tmp140;
}
}
}
}

auto tmp141 = make_vector<int64_t>( (int32_t)352);
/* Variable to read the clear value corresponding to the input variable tmp141 at (1385,1-1385,38) */
uint64_t __tmp_in_tmp141;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
cin >> __tmp_in_tmp141;
tmp141[i0] = __tmp_in_tmp141;
}

auto tmp142 = make_vector<int64_t>( (int32_t)352);
/* Variable to read the clear value corresponding to the input variable tmp142 at (1387,1-1387,38) */
uint64_t __tmp_in_tmp142;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
cin >> __tmp_in_tmp142;
tmp142[i0] = __tmp_in_tmp142;
}

auto tmp143 = make_vector<int64_t>( (int32_t)352);
/* Variable to read the clear value corresponding to the input variable tmp143 at (1389,1-1389,38) */
uint64_t __tmp_in_tmp143;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
cin >> __tmp_in_tmp143;
tmp143[i0] = __tmp_in_tmp143;
}

auto tmp144 = make_vector<int64_t>( (int32_t)352);
/* Variable to read the clear value corresponding to the input variable tmp144 at (1391,1-1391,38) */
uint64_t __tmp_in_tmp144;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
cin >> __tmp_in_tmp144;
tmp144[i0] = __tmp_in_tmp144;
}

auto tmp145 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)352,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp145 at (1393,1-1393,49) */
uint64_t __tmp_in_tmp145;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)352; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp145;
tmp145[i0][i1][i2][i3] = __tmp_in_tmp145;
}
}
}
}

auto tmp146 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp146 at (1395,1-1395,38) */
uint64_t __tmp_in_tmp146;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp146;
tmp146[i0] = __tmp_in_tmp146;
}

auto tmp147 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp147 at (1397,1-1397,38) */
uint64_t __tmp_in_tmp147;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp147;
tmp147[i0] = __tmp_in_tmp147;
}

auto tmp148 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp148 at (1399,1-1399,38) */
uint64_t __tmp_in_tmp148;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp148;
tmp148[i0] = __tmp_in_tmp148;
}

auto tmp149 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp149 at (1401,1-1401,38) */
uint64_t __tmp_in_tmp149;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp149;
tmp149[i0] = __tmp_in_tmp149;
}

auto tmp150 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp150 at (1403,1-1403,48) */
uint64_t __tmp_in_tmp150;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp150;
tmp150[i0][i1][i2][i3] = __tmp_in_tmp150;
}
}
}
}

auto tmp151 = make_vector<int64_t>( (int32_t)384);
/* Variable to read the clear value corresponding to the input variable tmp151 at (1405,1-1405,38) */
uint64_t __tmp_in_tmp151;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
cin >> __tmp_in_tmp151;
tmp151[i0] = __tmp_in_tmp151;
}

auto tmp152 = make_vector<int64_t>( (int32_t)384);
/* Variable to read the clear value corresponding to the input variable tmp152 at (1407,1-1407,38) */
uint64_t __tmp_in_tmp152;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
cin >> __tmp_in_tmp152;
tmp152[i0] = __tmp_in_tmp152;
}

auto tmp153 = make_vector<int64_t>( (int32_t)384);
/* Variable to read the clear value corresponding to the input variable tmp153 at (1409,1-1409,38) */
uint64_t __tmp_in_tmp153;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
cin >> __tmp_in_tmp153;
tmp153[i0] = __tmp_in_tmp153;
}

auto tmp154 = make_vector<int64_t>( (int32_t)384);
/* Variable to read the clear value corresponding to the input variable tmp154 at (1411,1-1411,38) */
uint64_t __tmp_in_tmp154;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
cin >> __tmp_in_tmp154;
tmp154[i0] = __tmp_in_tmp154;
}

auto tmp155 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp155 at (1413,1-1413,49) */
uint64_t __tmp_in_tmp155;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)384; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp155;
tmp155[i0][i1][i2][i3] = __tmp_in_tmp155;
}
}
}
}

auto tmp156 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp156 at (1415,1-1415,38) */
uint64_t __tmp_in_tmp156;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp156;
tmp156[i0] = __tmp_in_tmp156;
}

auto tmp157 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp157 at (1417,1-1417,38) */
uint64_t __tmp_in_tmp157;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp157;
tmp157[i0] = __tmp_in_tmp157;
}

auto tmp158 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp158 at (1419,1-1419,38) */
uint64_t __tmp_in_tmp158;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp158;
tmp158[i0] = __tmp_in_tmp158;
}

auto tmp159 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp159 at (1421,1-1421,38) */
uint64_t __tmp_in_tmp159;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp159;
tmp159[i0] = __tmp_in_tmp159;
}

auto tmp160 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp160 at (1423,1-1423,48) */
uint64_t __tmp_in_tmp160;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp160;
tmp160[i0][i1][i2][i3] = __tmp_in_tmp160;
}
}
}
}

auto tmp161 = make_vector<int64_t>( (int32_t)416);
/* Variable to read the clear value corresponding to the input variable tmp161 at (1425,1-1425,38) */
uint64_t __tmp_in_tmp161;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
cin >> __tmp_in_tmp161;
tmp161[i0] = __tmp_in_tmp161;
}

auto tmp162 = make_vector<int64_t>( (int32_t)416);
/* Variable to read the clear value corresponding to the input variable tmp162 at (1427,1-1427,38) */
uint64_t __tmp_in_tmp162;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
cin >> __tmp_in_tmp162;
tmp162[i0] = __tmp_in_tmp162;
}

auto tmp163 = make_vector<int64_t>( (int32_t)416);
/* Variable to read the clear value corresponding to the input variable tmp163 at (1429,1-1429,38) */
uint64_t __tmp_in_tmp163;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
cin >> __tmp_in_tmp163;
tmp163[i0] = __tmp_in_tmp163;
}

auto tmp164 = make_vector<int64_t>( (int32_t)416);
/* Variable to read the clear value corresponding to the input variable tmp164 at (1431,1-1431,38) */
uint64_t __tmp_in_tmp164;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
cin >> __tmp_in_tmp164;
tmp164[i0] = __tmp_in_tmp164;
}

auto tmp165 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)416,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp165 at (1433,1-1433,49) */
uint64_t __tmp_in_tmp165;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)416; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp165;
tmp165[i0][i1][i2][i3] = __tmp_in_tmp165;
}
}
}
}

auto tmp166 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp166 at (1435,1-1435,38) */
uint64_t __tmp_in_tmp166;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp166;
tmp166[i0] = __tmp_in_tmp166;
}

auto tmp167 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp167 at (1437,1-1437,38) */
uint64_t __tmp_in_tmp167;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp167;
tmp167[i0] = __tmp_in_tmp167;
}

auto tmp168 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp168 at (1439,1-1439,38) */
uint64_t __tmp_in_tmp168;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp168;
tmp168[i0] = __tmp_in_tmp168;
}

auto tmp169 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp169 at (1441,1-1441,38) */
uint64_t __tmp_in_tmp169;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp169;
tmp169[i0] = __tmp_in_tmp169;
}

auto tmp170 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp170 at (1443,1-1443,48) */
uint64_t __tmp_in_tmp170;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp170;
tmp170[i0][i1][i2][i3] = __tmp_in_tmp170;
}
}
}
}

auto tmp171 = make_vector<int64_t>( (int32_t)448);
/* Variable to read the clear value corresponding to the input variable tmp171 at (1445,1-1445,38) */
uint64_t __tmp_in_tmp171;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
cin >> __tmp_in_tmp171;
tmp171[i0] = __tmp_in_tmp171;
}

auto tmp172 = make_vector<int64_t>( (int32_t)448);
/* Variable to read the clear value corresponding to the input variable tmp172 at (1447,1-1447,38) */
uint64_t __tmp_in_tmp172;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
cin >> __tmp_in_tmp172;
tmp172[i0] = __tmp_in_tmp172;
}

auto tmp173 = make_vector<int64_t>( (int32_t)448);
/* Variable to read the clear value corresponding to the input variable tmp173 at (1449,1-1449,38) */
uint64_t __tmp_in_tmp173;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
cin >> __tmp_in_tmp173;
tmp173[i0] = __tmp_in_tmp173;
}

auto tmp174 = make_vector<int64_t>( (int32_t)448);
/* Variable to read the clear value corresponding to the input variable tmp174 at (1451,1-1451,38) */
uint64_t __tmp_in_tmp174;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
cin >> __tmp_in_tmp174;
tmp174[i0] = __tmp_in_tmp174;
}

auto tmp175 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)448,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp175 at (1453,1-1453,49) */
uint64_t __tmp_in_tmp175;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)448; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp175;
tmp175[i0][i1][i2][i3] = __tmp_in_tmp175;
}
}
}
}

auto tmp176 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp176 at (1455,1-1455,38) */
uint64_t __tmp_in_tmp176;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp176;
tmp176[i0] = __tmp_in_tmp176;
}

auto tmp177 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp177 at (1457,1-1457,38) */
uint64_t __tmp_in_tmp177;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp177;
tmp177[i0] = __tmp_in_tmp177;
}

auto tmp178 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp178 at (1459,1-1459,38) */
uint64_t __tmp_in_tmp178;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp178;
tmp178[i0] = __tmp_in_tmp178;
}

auto tmp179 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp179 at (1461,1-1461,38) */
uint64_t __tmp_in_tmp179;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp179;
tmp179[i0] = __tmp_in_tmp179;
}

auto tmp180 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp180 at (1463,1-1463,48) */
uint64_t __tmp_in_tmp180;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp180;
tmp180[i0][i1][i2][i3] = __tmp_in_tmp180;
}
}
}
}

auto tmp181 = make_vector<int64_t>( (int32_t)480);
/* Variable to read the clear value corresponding to the input variable tmp181 at (1465,1-1465,38) */
uint64_t __tmp_in_tmp181;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
cin >> __tmp_in_tmp181;
tmp181[i0] = __tmp_in_tmp181;
}

auto tmp182 = make_vector<int64_t>( (int32_t)480);
/* Variable to read the clear value corresponding to the input variable tmp182 at (1467,1-1467,38) */
uint64_t __tmp_in_tmp182;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
cin >> __tmp_in_tmp182;
tmp182[i0] = __tmp_in_tmp182;
}

auto tmp183 = make_vector<int64_t>( (int32_t)480);
/* Variable to read the clear value corresponding to the input variable tmp183 at (1469,1-1469,38) */
uint64_t __tmp_in_tmp183;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
cin >> __tmp_in_tmp183;
tmp183[i0] = __tmp_in_tmp183;
}

auto tmp184 = make_vector<int64_t>( (int32_t)480);
/* Variable to read the clear value corresponding to the input variable tmp184 at (1471,1-1471,38) */
uint64_t __tmp_in_tmp184;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
cin >> __tmp_in_tmp184;
tmp184[i0] = __tmp_in_tmp184;
}

auto tmp185 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)480,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp185 at (1473,1-1473,49) */
uint64_t __tmp_in_tmp185;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)480; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp185;
tmp185[i0][i1][i2][i3] = __tmp_in_tmp185;
}
}
}
}

auto tmp186 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp186 at (1475,1-1475,38) */
uint64_t __tmp_in_tmp186;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp186;
tmp186[i0] = __tmp_in_tmp186;
}

auto tmp187 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp187 at (1477,1-1477,38) */
uint64_t __tmp_in_tmp187;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp187;
tmp187[i0] = __tmp_in_tmp187;
}

auto tmp188 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp188 at (1479,1-1479,38) */
uint64_t __tmp_in_tmp188;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp188;
tmp188[i0] = __tmp_in_tmp188;
}

auto tmp189 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp189 at (1481,1-1481,38) */
uint64_t __tmp_in_tmp189;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp189;
tmp189[i0] = __tmp_in_tmp189;
}

auto tmp190 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp190 at (1483,1-1483,48) */
uint64_t __tmp_in_tmp190;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp190;
tmp190[i0][i1][i2][i3] = __tmp_in_tmp190;
}
}
}
}

auto tmp191 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp191 at (1485,1-1485,38) */
uint64_t __tmp_in_tmp191;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp191;
tmp191[i0] = __tmp_in_tmp191;
}

auto tmp192 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp192 at (1487,1-1487,38) */
uint64_t __tmp_in_tmp192;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp192;
tmp192[i0] = __tmp_in_tmp192;
}

auto tmp193 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp193 at (1489,1-1489,38) */
uint64_t __tmp_in_tmp193;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp193;
tmp193[i0] = __tmp_in_tmp193;
}

auto tmp194 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp194 at (1491,1-1491,38) */
uint64_t __tmp_in_tmp194;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp194;
tmp194[i0] = __tmp_in_tmp194;
}

auto tmp195 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp195 at (1493,1-1493,49) */
uint64_t __tmp_in_tmp195;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
cin >> __tmp_in_tmp195;
tmp195[i0][i1][i2][i3] = __tmp_in_tmp195;
}
}
}
}

auto tmp196 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp196 at (1495,1-1495,38) */
uint64_t __tmp_in_tmp196;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp196;
tmp196[i0] = __tmp_in_tmp196;
}

auto tmp197 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp197 at (1497,1-1497,38) */
uint64_t __tmp_in_tmp197;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp197;
tmp197[i0] = __tmp_in_tmp197;
}

auto tmp198 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp198 at (1499,1-1499,38) */
uint64_t __tmp_in_tmp198;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp198;
tmp198[i0] = __tmp_in_tmp198;
}

auto tmp199 = make_vector<int64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp199 at (1501,1-1501,38) */
uint64_t __tmp_in_tmp199;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
cin >> __tmp_in_tmp199;
tmp199[i0] = __tmp_in_tmp199;
}

auto tmp200 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp200 at (1503,1-1503,49) */
uint64_t __tmp_in_tmp200;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp200;
tmp200[i0][i1][i2][i3] = __tmp_in_tmp200;
}
}
}
}

auto tmp201 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp201 at (1505,1-1505,38) */
uint64_t __tmp_in_tmp201;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp201;
tmp201[i0] = __tmp_in_tmp201;
}

auto tmp202 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp202 at (1507,1-1507,38) */
uint64_t __tmp_in_tmp202;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp202;
tmp202[i0] = __tmp_in_tmp202;
}

auto tmp203 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp203 at (1509,1-1509,38) */
uint64_t __tmp_in_tmp203;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp203;
tmp203[i0] = __tmp_in_tmp203;
}

auto tmp204 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp204 at (1511,1-1511,38) */
uint64_t __tmp_in_tmp204;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp204;
tmp204[i0] = __tmp_in_tmp204;
}

auto tmp205 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp205 at (1513,1-1513,48) */
uint64_t __tmp_in_tmp205;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp205;
tmp205[i0][i1][i2][i3] = __tmp_in_tmp205;
}
}
}
}

auto tmp206 = make_vector<int64_t>( (int32_t)288);
/* Variable to read the clear value corresponding to the input variable tmp206 at (1515,1-1515,38) */
uint64_t __tmp_in_tmp206;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
cin >> __tmp_in_tmp206;
tmp206[i0] = __tmp_in_tmp206;
}

auto tmp207 = make_vector<int64_t>( (int32_t)288);
/* Variable to read the clear value corresponding to the input variable tmp207 at (1517,1-1517,38) */
uint64_t __tmp_in_tmp207;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
cin >> __tmp_in_tmp207;
tmp207[i0] = __tmp_in_tmp207;
}

auto tmp208 = make_vector<int64_t>( (int32_t)288);
/* Variable to read the clear value corresponding to the input variable tmp208 at (1519,1-1519,38) */
uint64_t __tmp_in_tmp208;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
cin >> __tmp_in_tmp208;
tmp208[i0] = __tmp_in_tmp208;
}

auto tmp209 = make_vector<int64_t>( (int32_t)288);
/* Variable to read the clear value corresponding to the input variable tmp209 at (1521,1-1521,38) */
uint64_t __tmp_in_tmp209;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
cin >> __tmp_in_tmp209;
tmp209[i0] = __tmp_in_tmp209;
}

auto tmp210 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)288,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp210 at (1523,1-1523,49) */
uint64_t __tmp_in_tmp210;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)288; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp210;
tmp210[i0][i1][i2][i3] = __tmp_in_tmp210;
}
}
}
}

auto tmp211 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp211 at (1525,1-1525,38) */
uint64_t __tmp_in_tmp211;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp211;
tmp211[i0] = __tmp_in_tmp211;
}

auto tmp212 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp212 at (1527,1-1527,38) */
uint64_t __tmp_in_tmp212;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp212;
tmp212[i0] = __tmp_in_tmp212;
}

auto tmp213 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp213 at (1529,1-1529,38) */
uint64_t __tmp_in_tmp213;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp213;
tmp213[i0] = __tmp_in_tmp213;
}

auto tmp214 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp214 at (1531,1-1531,38) */
uint64_t __tmp_in_tmp214;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp214;
tmp214[i0] = __tmp_in_tmp214;
}

auto tmp215 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp215 at (1533,1-1533,48) */
uint64_t __tmp_in_tmp215;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp215;
tmp215[i0][i1][i2][i3] = __tmp_in_tmp215;
}
}
}
}

auto tmp216 = make_vector<int64_t>( (int32_t)320);
/* Variable to read the clear value corresponding to the input variable tmp216 at (1535,1-1535,38) */
uint64_t __tmp_in_tmp216;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
cin >> __tmp_in_tmp216;
tmp216[i0] = __tmp_in_tmp216;
}

auto tmp217 = make_vector<int64_t>( (int32_t)320);
/* Variable to read the clear value corresponding to the input variable tmp217 at (1537,1-1537,38) */
uint64_t __tmp_in_tmp217;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
cin >> __tmp_in_tmp217;
tmp217[i0] = __tmp_in_tmp217;
}

auto tmp218 = make_vector<int64_t>( (int32_t)320);
/* Variable to read the clear value corresponding to the input variable tmp218 at (1539,1-1539,38) */
uint64_t __tmp_in_tmp218;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
cin >> __tmp_in_tmp218;
tmp218[i0] = __tmp_in_tmp218;
}

auto tmp219 = make_vector<int64_t>( (int32_t)320);
/* Variable to read the clear value corresponding to the input variable tmp219 at (1541,1-1541,38) */
uint64_t __tmp_in_tmp219;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
cin >> __tmp_in_tmp219;
tmp219[i0] = __tmp_in_tmp219;
}

auto tmp220 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)320,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp220 at (1543,1-1543,49) */
uint64_t __tmp_in_tmp220;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)320; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp220;
tmp220[i0][i1][i2][i3] = __tmp_in_tmp220;
}
}
}
}

auto tmp221 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp221 at (1545,1-1545,38) */
uint64_t __tmp_in_tmp221;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp221;
tmp221[i0] = __tmp_in_tmp221;
}

auto tmp222 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp222 at (1547,1-1547,38) */
uint64_t __tmp_in_tmp222;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp222;
tmp222[i0] = __tmp_in_tmp222;
}

auto tmp223 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp223 at (1549,1-1549,38) */
uint64_t __tmp_in_tmp223;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp223;
tmp223[i0] = __tmp_in_tmp223;
}

auto tmp224 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp224 at (1551,1-1551,38) */
uint64_t __tmp_in_tmp224;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp224;
tmp224[i0] = __tmp_in_tmp224;
}

auto tmp225 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp225 at (1553,1-1553,48) */
uint64_t __tmp_in_tmp225;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp225;
tmp225[i0][i1][i2][i3] = __tmp_in_tmp225;
}
}
}
}

auto tmp226 = make_vector<int64_t>( (int32_t)352);
/* Variable to read the clear value corresponding to the input variable tmp226 at (1555,1-1555,38) */
uint64_t __tmp_in_tmp226;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
cin >> __tmp_in_tmp226;
tmp226[i0] = __tmp_in_tmp226;
}

auto tmp227 = make_vector<int64_t>( (int32_t)352);
/* Variable to read the clear value corresponding to the input variable tmp227 at (1557,1-1557,38) */
uint64_t __tmp_in_tmp227;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
cin >> __tmp_in_tmp227;
tmp227[i0] = __tmp_in_tmp227;
}

auto tmp228 = make_vector<int64_t>( (int32_t)352);
/* Variable to read the clear value corresponding to the input variable tmp228 at (1559,1-1559,38) */
uint64_t __tmp_in_tmp228;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
cin >> __tmp_in_tmp228;
tmp228[i0] = __tmp_in_tmp228;
}

auto tmp229 = make_vector<int64_t>( (int32_t)352);
/* Variable to read the clear value corresponding to the input variable tmp229 at (1561,1-1561,38) */
uint64_t __tmp_in_tmp229;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
cin >> __tmp_in_tmp229;
tmp229[i0] = __tmp_in_tmp229;
}

auto tmp230 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)352,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp230 at (1563,1-1563,49) */
uint64_t __tmp_in_tmp230;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)352; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp230;
tmp230[i0][i1][i2][i3] = __tmp_in_tmp230;
}
}
}
}

auto tmp231 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp231 at (1565,1-1565,38) */
uint64_t __tmp_in_tmp231;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp231;
tmp231[i0] = __tmp_in_tmp231;
}

auto tmp232 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp232 at (1567,1-1567,38) */
uint64_t __tmp_in_tmp232;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp232;
tmp232[i0] = __tmp_in_tmp232;
}

auto tmp233 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp233 at (1569,1-1569,38) */
uint64_t __tmp_in_tmp233;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp233;
tmp233[i0] = __tmp_in_tmp233;
}

auto tmp234 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp234 at (1571,1-1571,38) */
uint64_t __tmp_in_tmp234;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp234;
tmp234[i0] = __tmp_in_tmp234;
}

auto tmp235 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp235 at (1573,1-1573,48) */
uint64_t __tmp_in_tmp235;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp235;
tmp235[i0][i1][i2][i3] = __tmp_in_tmp235;
}
}
}
}

auto tmp236 = make_vector<int64_t>( (int32_t)384);
/* Variable to read the clear value corresponding to the input variable tmp236 at (1575,1-1575,38) */
uint64_t __tmp_in_tmp236;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
cin >> __tmp_in_tmp236;
tmp236[i0] = __tmp_in_tmp236;
}

auto tmp237 = make_vector<int64_t>( (int32_t)384);
/* Variable to read the clear value corresponding to the input variable tmp237 at (1577,1-1577,38) */
uint64_t __tmp_in_tmp237;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
cin >> __tmp_in_tmp237;
tmp237[i0] = __tmp_in_tmp237;
}

auto tmp238 = make_vector<int64_t>( (int32_t)384);
/* Variable to read the clear value corresponding to the input variable tmp238 at (1579,1-1579,38) */
uint64_t __tmp_in_tmp238;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
cin >> __tmp_in_tmp238;
tmp238[i0] = __tmp_in_tmp238;
}

auto tmp239 = make_vector<int64_t>( (int32_t)384);
/* Variable to read the clear value corresponding to the input variable tmp239 at (1581,1-1581,38) */
uint64_t __tmp_in_tmp239;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
cin >> __tmp_in_tmp239;
tmp239[i0] = __tmp_in_tmp239;
}

auto tmp240 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp240 at (1583,1-1583,49) */
uint64_t __tmp_in_tmp240;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)384; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp240;
tmp240[i0][i1][i2][i3] = __tmp_in_tmp240;
}
}
}
}

auto tmp241 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp241 at (1585,1-1585,38) */
uint64_t __tmp_in_tmp241;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp241;
tmp241[i0] = __tmp_in_tmp241;
}

auto tmp242 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp242 at (1587,1-1587,38) */
uint64_t __tmp_in_tmp242;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp242;
tmp242[i0] = __tmp_in_tmp242;
}

auto tmp243 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp243 at (1589,1-1589,38) */
uint64_t __tmp_in_tmp243;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp243;
tmp243[i0] = __tmp_in_tmp243;
}

auto tmp244 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp244 at (1591,1-1591,38) */
uint64_t __tmp_in_tmp244;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp244;
tmp244[i0] = __tmp_in_tmp244;
}

auto tmp245 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp245 at (1593,1-1593,48) */
uint64_t __tmp_in_tmp245;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp245;
tmp245[i0][i1][i2][i3] = __tmp_in_tmp245;
}
}
}
}

auto tmp246 = make_vector<int64_t>( (int32_t)416);
/* Variable to read the clear value corresponding to the input variable tmp246 at (1595,1-1595,38) */
uint64_t __tmp_in_tmp246;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
cin >> __tmp_in_tmp246;
tmp246[i0] = __tmp_in_tmp246;
}

auto tmp247 = make_vector<int64_t>( (int32_t)416);
/* Variable to read the clear value corresponding to the input variable tmp247 at (1597,1-1597,38) */
uint64_t __tmp_in_tmp247;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
cin >> __tmp_in_tmp247;
tmp247[i0] = __tmp_in_tmp247;
}

auto tmp248 = make_vector<int64_t>( (int32_t)416);
/* Variable to read the clear value corresponding to the input variable tmp248 at (1599,1-1599,38) */
uint64_t __tmp_in_tmp248;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
cin >> __tmp_in_tmp248;
tmp248[i0] = __tmp_in_tmp248;
}

auto tmp249 = make_vector<int64_t>( (int32_t)416);
/* Variable to read the clear value corresponding to the input variable tmp249 at (1601,1-1601,38) */
uint64_t __tmp_in_tmp249;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
cin >> __tmp_in_tmp249;
tmp249[i0] = __tmp_in_tmp249;
}

auto tmp250 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)416,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp250 at (1603,1-1603,49) */
uint64_t __tmp_in_tmp250;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)416; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp250;
tmp250[i0][i1][i2][i3] = __tmp_in_tmp250;
}
}
}
}

auto tmp251 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp251 at (1605,1-1605,38) */
uint64_t __tmp_in_tmp251;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp251;
tmp251[i0] = __tmp_in_tmp251;
}

auto tmp252 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp252 at (1607,1-1607,38) */
uint64_t __tmp_in_tmp252;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp252;
tmp252[i0] = __tmp_in_tmp252;
}

auto tmp253 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp253 at (1609,1-1609,38) */
uint64_t __tmp_in_tmp253;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp253;
tmp253[i0] = __tmp_in_tmp253;
}

auto tmp254 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp254 at (1611,1-1611,38) */
uint64_t __tmp_in_tmp254;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp254;
tmp254[i0] = __tmp_in_tmp254;
}

auto tmp255 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp255 at (1613,1-1613,48) */
uint64_t __tmp_in_tmp255;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp255;
tmp255[i0][i1][i2][i3] = __tmp_in_tmp255;
}
}
}
}

auto tmp256 = make_vector<int64_t>( (int32_t)448);
/* Variable to read the clear value corresponding to the input variable tmp256 at (1615,1-1615,38) */
uint64_t __tmp_in_tmp256;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
cin >> __tmp_in_tmp256;
tmp256[i0] = __tmp_in_tmp256;
}

auto tmp257 = make_vector<int64_t>( (int32_t)448);
/* Variable to read the clear value corresponding to the input variable tmp257 at (1617,1-1617,38) */
uint64_t __tmp_in_tmp257;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
cin >> __tmp_in_tmp257;
tmp257[i0] = __tmp_in_tmp257;
}

auto tmp258 = make_vector<int64_t>( (int32_t)448);
/* Variable to read the clear value corresponding to the input variable tmp258 at (1619,1-1619,38) */
uint64_t __tmp_in_tmp258;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
cin >> __tmp_in_tmp258;
tmp258[i0] = __tmp_in_tmp258;
}

auto tmp259 = make_vector<int64_t>( (int32_t)448);
/* Variable to read the clear value corresponding to the input variable tmp259 at (1621,1-1621,38) */
uint64_t __tmp_in_tmp259;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
cin >> __tmp_in_tmp259;
tmp259[i0] = __tmp_in_tmp259;
}

auto tmp260 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)448,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp260 at (1623,1-1623,49) */
uint64_t __tmp_in_tmp260;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)448; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp260;
tmp260[i0][i1][i2][i3] = __tmp_in_tmp260;
}
}
}
}

auto tmp261 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp261 at (1625,1-1625,38) */
uint64_t __tmp_in_tmp261;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp261;
tmp261[i0] = __tmp_in_tmp261;
}

auto tmp262 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp262 at (1627,1-1627,38) */
uint64_t __tmp_in_tmp262;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp262;
tmp262[i0] = __tmp_in_tmp262;
}

auto tmp263 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp263 at (1629,1-1629,38) */
uint64_t __tmp_in_tmp263;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp263;
tmp263[i0] = __tmp_in_tmp263;
}

auto tmp264 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp264 at (1631,1-1631,38) */
uint64_t __tmp_in_tmp264;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp264;
tmp264[i0] = __tmp_in_tmp264;
}

auto tmp265 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp265 at (1633,1-1633,48) */
uint64_t __tmp_in_tmp265;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp265;
tmp265[i0][i1][i2][i3] = __tmp_in_tmp265;
}
}
}
}

auto tmp266 = make_vector<int64_t>( (int32_t)480);
/* Variable to read the clear value corresponding to the input variable tmp266 at (1635,1-1635,38) */
uint64_t __tmp_in_tmp266;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
cin >> __tmp_in_tmp266;
tmp266[i0] = __tmp_in_tmp266;
}

auto tmp267 = make_vector<int64_t>( (int32_t)480);
/* Variable to read the clear value corresponding to the input variable tmp267 at (1637,1-1637,38) */
uint64_t __tmp_in_tmp267;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
cin >> __tmp_in_tmp267;
tmp267[i0] = __tmp_in_tmp267;
}

auto tmp268 = make_vector<int64_t>( (int32_t)480);
/* Variable to read the clear value corresponding to the input variable tmp268 at (1639,1-1639,38) */
uint64_t __tmp_in_tmp268;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
cin >> __tmp_in_tmp268;
tmp268[i0] = __tmp_in_tmp268;
}

auto tmp269 = make_vector<int64_t>( (int32_t)480);
/* Variable to read the clear value corresponding to the input variable tmp269 at (1641,1-1641,38) */
uint64_t __tmp_in_tmp269;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
cin >> __tmp_in_tmp269;
tmp269[i0] = __tmp_in_tmp269;
}

auto tmp270 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)480,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp270 at (1643,1-1643,49) */
uint64_t __tmp_in_tmp270;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)480; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp270;
tmp270[i0][i1][i2][i3] = __tmp_in_tmp270;
}
}
}
}

auto tmp271 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp271 at (1645,1-1645,38) */
uint64_t __tmp_in_tmp271;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp271;
tmp271[i0] = __tmp_in_tmp271;
}

auto tmp272 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp272 at (1647,1-1647,38) */
uint64_t __tmp_in_tmp272;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp272;
tmp272[i0] = __tmp_in_tmp272;
}

auto tmp273 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp273 at (1649,1-1649,38) */
uint64_t __tmp_in_tmp273;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp273;
tmp273[i0] = __tmp_in_tmp273;
}

auto tmp274 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp274 at (1651,1-1651,38) */
uint64_t __tmp_in_tmp274;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp274;
tmp274[i0] = __tmp_in_tmp274;
}

auto tmp275 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp275 at (1653,1-1653,48) */
uint64_t __tmp_in_tmp275;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp275;
tmp275[i0][i1][i2][i3] = __tmp_in_tmp275;
}
}
}
}

auto tmp276 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp276 at (1655,1-1655,38) */
uint64_t __tmp_in_tmp276;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp276;
tmp276[i0] = __tmp_in_tmp276;
}

auto tmp277 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp277 at (1657,1-1657,38) */
uint64_t __tmp_in_tmp277;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp277;
tmp277[i0] = __tmp_in_tmp277;
}

auto tmp278 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp278 at (1659,1-1659,38) */
uint64_t __tmp_in_tmp278;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp278;
tmp278[i0] = __tmp_in_tmp278;
}

auto tmp279 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp279 at (1661,1-1661,38) */
uint64_t __tmp_in_tmp279;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp279;
tmp279[i0] = __tmp_in_tmp279;
}

auto tmp280 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp280 at (1663,1-1663,49) */
uint64_t __tmp_in_tmp280;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp280;
tmp280[i0][i1][i2][i3] = __tmp_in_tmp280;
}
}
}
}

auto tmp281 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp281 at (1665,1-1665,38) */
uint64_t __tmp_in_tmp281;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp281;
tmp281[i0] = __tmp_in_tmp281;
}

auto tmp282 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp282 at (1667,1-1667,38) */
uint64_t __tmp_in_tmp282;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp282;
tmp282[i0] = __tmp_in_tmp282;
}

auto tmp283 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp283 at (1669,1-1669,38) */
uint64_t __tmp_in_tmp283;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp283;
tmp283[i0] = __tmp_in_tmp283;
}

auto tmp284 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp284 at (1671,1-1671,38) */
uint64_t __tmp_in_tmp284;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp284;
tmp284[i0] = __tmp_in_tmp284;
}

auto tmp285 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp285 at (1673,1-1673,48) */
uint64_t __tmp_in_tmp285;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp285;
tmp285[i0][i1][i2][i3] = __tmp_in_tmp285;
}
}
}
}

auto tmp286 = make_vector<int64_t>( (int32_t)544);
/* Variable to read the clear value corresponding to the input variable tmp286 at (1675,1-1675,38) */
uint64_t __tmp_in_tmp286;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
cin >> __tmp_in_tmp286;
tmp286[i0] = __tmp_in_tmp286;
}

auto tmp287 = make_vector<int64_t>( (int32_t)544);
/* Variable to read the clear value corresponding to the input variable tmp287 at (1677,1-1677,38) */
uint64_t __tmp_in_tmp287;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
cin >> __tmp_in_tmp287;
tmp287[i0] = __tmp_in_tmp287;
}

auto tmp288 = make_vector<int64_t>( (int32_t)544);
/* Variable to read the clear value corresponding to the input variable tmp288 at (1679,1-1679,38) */
uint64_t __tmp_in_tmp288;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
cin >> __tmp_in_tmp288;
tmp288[i0] = __tmp_in_tmp288;
}

auto tmp289 = make_vector<int64_t>( (int32_t)544);
/* Variable to read the clear value corresponding to the input variable tmp289 at (1681,1-1681,38) */
uint64_t __tmp_in_tmp289;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
cin >> __tmp_in_tmp289;
tmp289[i0] = __tmp_in_tmp289;
}

auto tmp290 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)544,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp290 at (1683,1-1683,49) */
uint64_t __tmp_in_tmp290;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)544; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp290;
tmp290[i0][i1][i2][i3] = __tmp_in_tmp290;
}
}
}
}

auto tmp291 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp291 at (1685,1-1685,38) */
uint64_t __tmp_in_tmp291;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp291;
tmp291[i0] = __tmp_in_tmp291;
}

auto tmp292 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp292 at (1687,1-1687,38) */
uint64_t __tmp_in_tmp292;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp292;
tmp292[i0] = __tmp_in_tmp292;
}

auto tmp293 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp293 at (1689,1-1689,38) */
uint64_t __tmp_in_tmp293;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp293;
tmp293[i0] = __tmp_in_tmp293;
}

auto tmp294 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp294 at (1691,1-1691,38) */
uint64_t __tmp_in_tmp294;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp294;
tmp294[i0] = __tmp_in_tmp294;
}

auto tmp295 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp295 at (1693,1-1693,48) */
uint64_t __tmp_in_tmp295;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp295;
tmp295[i0][i1][i2][i3] = __tmp_in_tmp295;
}
}
}
}

auto tmp296 = make_vector<int64_t>( (int32_t)576);
/* Variable to read the clear value corresponding to the input variable tmp296 at (1695,1-1695,38) */
uint64_t __tmp_in_tmp296;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
cin >> __tmp_in_tmp296;
tmp296[i0] = __tmp_in_tmp296;
}

auto tmp297 = make_vector<int64_t>( (int32_t)576);
/* Variable to read the clear value corresponding to the input variable tmp297 at (1697,1-1697,38) */
uint64_t __tmp_in_tmp297;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
cin >> __tmp_in_tmp297;
tmp297[i0] = __tmp_in_tmp297;
}

auto tmp298 = make_vector<int64_t>( (int32_t)576);
/* Variable to read the clear value corresponding to the input variable tmp298 at (1699,1-1699,38) */
uint64_t __tmp_in_tmp298;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
cin >> __tmp_in_tmp298;
tmp298[i0] = __tmp_in_tmp298;
}

auto tmp299 = make_vector<int64_t>( (int32_t)576);
/* Variable to read the clear value corresponding to the input variable tmp299 at (1701,1-1701,38) */
uint64_t __tmp_in_tmp299;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
cin >> __tmp_in_tmp299;
tmp299[i0] = __tmp_in_tmp299;
}

auto tmp300 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)576,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp300 at (1703,1-1703,49) */
uint64_t __tmp_in_tmp300;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)576; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp300;
tmp300[i0][i1][i2][i3] = __tmp_in_tmp300;
}
}
}
}

auto tmp301 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp301 at (1705,1-1705,38) */
uint64_t __tmp_in_tmp301;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp301;
tmp301[i0] = __tmp_in_tmp301;
}

auto tmp302 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp302 at (1707,1-1707,38) */
uint64_t __tmp_in_tmp302;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp302;
tmp302[i0] = __tmp_in_tmp302;
}

auto tmp303 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp303 at (1709,1-1709,38) */
uint64_t __tmp_in_tmp303;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp303;
tmp303[i0] = __tmp_in_tmp303;
}

auto tmp304 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp304 at (1711,1-1711,38) */
uint64_t __tmp_in_tmp304;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp304;
tmp304[i0] = __tmp_in_tmp304;
}

auto tmp305 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp305 at (1713,1-1713,48) */
uint64_t __tmp_in_tmp305;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp305;
tmp305[i0][i1][i2][i3] = __tmp_in_tmp305;
}
}
}
}

auto tmp306 = make_vector<int64_t>( (int32_t)608);
/* Variable to read the clear value corresponding to the input variable tmp306 at (1715,1-1715,38) */
uint64_t __tmp_in_tmp306;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
cin >> __tmp_in_tmp306;
tmp306[i0] = __tmp_in_tmp306;
}

auto tmp307 = make_vector<int64_t>( (int32_t)608);
/* Variable to read the clear value corresponding to the input variable tmp307 at (1717,1-1717,38) */
uint64_t __tmp_in_tmp307;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
cin >> __tmp_in_tmp307;
tmp307[i0] = __tmp_in_tmp307;
}

auto tmp308 = make_vector<int64_t>( (int32_t)608);
/* Variable to read the clear value corresponding to the input variable tmp308 at (1719,1-1719,38) */
uint64_t __tmp_in_tmp308;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
cin >> __tmp_in_tmp308;
tmp308[i0] = __tmp_in_tmp308;
}

auto tmp309 = make_vector<int64_t>( (int32_t)608);
/* Variable to read the clear value corresponding to the input variable tmp309 at (1721,1-1721,38) */
uint64_t __tmp_in_tmp309;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
cin >> __tmp_in_tmp309;
tmp309[i0] = __tmp_in_tmp309;
}

auto tmp310 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)608,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp310 at (1723,1-1723,49) */
uint64_t __tmp_in_tmp310;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)608; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp310;
tmp310[i0][i1][i2][i3] = __tmp_in_tmp310;
}
}
}
}

auto tmp311 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp311 at (1725,1-1725,38) */
uint64_t __tmp_in_tmp311;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp311;
tmp311[i0] = __tmp_in_tmp311;
}

auto tmp312 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp312 at (1727,1-1727,38) */
uint64_t __tmp_in_tmp312;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp312;
tmp312[i0] = __tmp_in_tmp312;
}

auto tmp313 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp313 at (1729,1-1729,38) */
uint64_t __tmp_in_tmp313;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp313;
tmp313[i0] = __tmp_in_tmp313;
}

auto tmp314 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp314 at (1731,1-1731,38) */
uint64_t __tmp_in_tmp314;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp314;
tmp314[i0] = __tmp_in_tmp314;
}

auto tmp315 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp315 at (1733,1-1733,48) */
uint64_t __tmp_in_tmp315;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp315;
tmp315[i0][i1][i2][i3] = __tmp_in_tmp315;
}
}
}
}

auto tmp316 = make_vector<int64_t>( (int32_t)640);
/* Variable to read the clear value corresponding to the input variable tmp316 at (1735,1-1735,38) */
uint64_t __tmp_in_tmp316;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
cin >> __tmp_in_tmp316;
tmp316[i0] = __tmp_in_tmp316;
}

auto tmp317 = make_vector<int64_t>( (int32_t)640);
/* Variable to read the clear value corresponding to the input variable tmp317 at (1737,1-1737,38) */
uint64_t __tmp_in_tmp317;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
cin >> __tmp_in_tmp317;
tmp317[i0] = __tmp_in_tmp317;
}

auto tmp318 = make_vector<int64_t>( (int32_t)640);
/* Variable to read the clear value corresponding to the input variable tmp318 at (1739,1-1739,38) */
uint64_t __tmp_in_tmp318;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
cin >> __tmp_in_tmp318;
tmp318[i0] = __tmp_in_tmp318;
}

auto tmp319 = make_vector<int64_t>( (int32_t)640);
/* Variable to read the clear value corresponding to the input variable tmp319 at (1741,1-1741,38) */
uint64_t __tmp_in_tmp319;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
cin >> __tmp_in_tmp319;
tmp319[i0] = __tmp_in_tmp319;
}

auto tmp320 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)640,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp320 at (1743,1-1743,49) */
uint64_t __tmp_in_tmp320;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)640; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp320;
tmp320[i0][i1][i2][i3] = __tmp_in_tmp320;
}
}
}
}

auto tmp321 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp321 at (1745,1-1745,38) */
uint64_t __tmp_in_tmp321;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp321;
tmp321[i0] = __tmp_in_tmp321;
}

auto tmp322 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp322 at (1747,1-1747,38) */
uint64_t __tmp_in_tmp322;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp322;
tmp322[i0] = __tmp_in_tmp322;
}

auto tmp323 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp323 at (1749,1-1749,38) */
uint64_t __tmp_in_tmp323;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp323;
tmp323[i0] = __tmp_in_tmp323;
}

auto tmp324 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp324 at (1751,1-1751,38) */
uint64_t __tmp_in_tmp324;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp324;
tmp324[i0] = __tmp_in_tmp324;
}

auto tmp325 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp325 at (1753,1-1753,48) */
uint64_t __tmp_in_tmp325;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp325;
tmp325[i0][i1][i2][i3] = __tmp_in_tmp325;
}
}
}
}

auto tmp326 = make_vector<int64_t>( (int32_t)672);
/* Variable to read the clear value corresponding to the input variable tmp326 at (1755,1-1755,38) */
uint64_t __tmp_in_tmp326;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
cin >> __tmp_in_tmp326;
tmp326[i0] = __tmp_in_tmp326;
}

auto tmp327 = make_vector<int64_t>( (int32_t)672);
/* Variable to read the clear value corresponding to the input variable tmp327 at (1757,1-1757,38) */
uint64_t __tmp_in_tmp327;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
cin >> __tmp_in_tmp327;
tmp327[i0] = __tmp_in_tmp327;
}

auto tmp328 = make_vector<int64_t>( (int32_t)672);
/* Variable to read the clear value corresponding to the input variable tmp328 at (1759,1-1759,38) */
uint64_t __tmp_in_tmp328;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
cin >> __tmp_in_tmp328;
tmp328[i0] = __tmp_in_tmp328;
}

auto tmp329 = make_vector<int64_t>( (int32_t)672);
/* Variable to read the clear value corresponding to the input variable tmp329 at (1761,1-1761,38) */
uint64_t __tmp_in_tmp329;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
cin >> __tmp_in_tmp329;
tmp329[i0] = __tmp_in_tmp329;
}

auto tmp330 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)672,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp330 at (1763,1-1763,49) */
uint64_t __tmp_in_tmp330;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)672; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp330;
tmp330[i0][i1][i2][i3] = __tmp_in_tmp330;
}
}
}
}

auto tmp331 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp331 at (1765,1-1765,38) */
uint64_t __tmp_in_tmp331;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp331;
tmp331[i0] = __tmp_in_tmp331;
}

auto tmp332 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp332 at (1767,1-1767,38) */
uint64_t __tmp_in_tmp332;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp332;
tmp332[i0] = __tmp_in_tmp332;
}

auto tmp333 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp333 at (1769,1-1769,38) */
uint64_t __tmp_in_tmp333;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp333;
tmp333[i0] = __tmp_in_tmp333;
}

auto tmp334 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp334 at (1771,1-1771,38) */
uint64_t __tmp_in_tmp334;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp334;
tmp334[i0] = __tmp_in_tmp334;
}

auto tmp335 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp335 at (1773,1-1773,48) */
uint64_t __tmp_in_tmp335;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp335;
tmp335[i0][i1][i2][i3] = __tmp_in_tmp335;
}
}
}
}

auto tmp336 = make_vector<int64_t>( (int32_t)704);
/* Variable to read the clear value corresponding to the input variable tmp336 at (1775,1-1775,38) */
uint64_t __tmp_in_tmp336;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
cin >> __tmp_in_tmp336;
tmp336[i0] = __tmp_in_tmp336;
}

auto tmp337 = make_vector<int64_t>( (int32_t)704);
/* Variable to read the clear value corresponding to the input variable tmp337 at (1777,1-1777,38) */
uint64_t __tmp_in_tmp337;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
cin >> __tmp_in_tmp337;
tmp337[i0] = __tmp_in_tmp337;
}

auto tmp338 = make_vector<int64_t>( (int32_t)704);
/* Variable to read the clear value corresponding to the input variable tmp338 at (1779,1-1779,38) */
uint64_t __tmp_in_tmp338;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
cin >> __tmp_in_tmp338;
tmp338[i0] = __tmp_in_tmp338;
}

auto tmp339 = make_vector<int64_t>( (int32_t)704);
/* Variable to read the clear value corresponding to the input variable tmp339 at (1781,1-1781,38) */
uint64_t __tmp_in_tmp339;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
cin >> __tmp_in_tmp339;
tmp339[i0] = __tmp_in_tmp339;
}

auto tmp340 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)704,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp340 at (1783,1-1783,49) */
uint64_t __tmp_in_tmp340;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)704; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp340;
tmp340[i0][i1][i2][i3] = __tmp_in_tmp340;
}
}
}
}

auto tmp341 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp341 at (1785,1-1785,38) */
uint64_t __tmp_in_tmp341;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp341;
tmp341[i0] = __tmp_in_tmp341;
}

auto tmp342 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp342 at (1787,1-1787,38) */
uint64_t __tmp_in_tmp342;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp342;
tmp342[i0] = __tmp_in_tmp342;
}

auto tmp343 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp343 at (1789,1-1789,38) */
uint64_t __tmp_in_tmp343;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp343;
tmp343[i0] = __tmp_in_tmp343;
}

auto tmp344 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp344 at (1791,1-1791,38) */
uint64_t __tmp_in_tmp344;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp344;
tmp344[i0] = __tmp_in_tmp344;
}

auto tmp345 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp345 at (1793,1-1793,48) */
uint64_t __tmp_in_tmp345;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp345;
tmp345[i0][i1][i2][i3] = __tmp_in_tmp345;
}
}
}
}

auto tmp346 = make_vector<int64_t>( (int32_t)736);
/* Variable to read the clear value corresponding to the input variable tmp346 at (1795,1-1795,38) */
uint64_t __tmp_in_tmp346;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
cin >> __tmp_in_tmp346;
tmp346[i0] = __tmp_in_tmp346;
}

auto tmp347 = make_vector<int64_t>( (int32_t)736);
/* Variable to read the clear value corresponding to the input variable tmp347 at (1797,1-1797,38) */
uint64_t __tmp_in_tmp347;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
cin >> __tmp_in_tmp347;
tmp347[i0] = __tmp_in_tmp347;
}

auto tmp348 = make_vector<int64_t>( (int32_t)736);
/* Variable to read the clear value corresponding to the input variable tmp348 at (1799,1-1799,38) */
uint64_t __tmp_in_tmp348;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
cin >> __tmp_in_tmp348;
tmp348[i0] = __tmp_in_tmp348;
}

auto tmp349 = make_vector<int64_t>( (int32_t)736);
/* Variable to read the clear value corresponding to the input variable tmp349 at (1801,1-1801,38) */
uint64_t __tmp_in_tmp349;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
cin >> __tmp_in_tmp349;
tmp349[i0] = __tmp_in_tmp349;
}

auto tmp350 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)736,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp350 at (1803,1-1803,49) */
uint64_t __tmp_in_tmp350;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)736; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp350;
tmp350[i0][i1][i2][i3] = __tmp_in_tmp350;
}
}
}
}

auto tmp351 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp351 at (1805,1-1805,38) */
uint64_t __tmp_in_tmp351;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp351;
tmp351[i0] = __tmp_in_tmp351;
}

auto tmp352 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp352 at (1807,1-1807,38) */
uint64_t __tmp_in_tmp352;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp352;
tmp352[i0] = __tmp_in_tmp352;
}

auto tmp353 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp353 at (1809,1-1809,38) */
uint64_t __tmp_in_tmp353;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp353;
tmp353[i0] = __tmp_in_tmp353;
}

auto tmp354 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp354 at (1811,1-1811,38) */
uint64_t __tmp_in_tmp354;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp354;
tmp354[i0] = __tmp_in_tmp354;
}

auto tmp355 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp355 at (1813,1-1813,48) */
uint64_t __tmp_in_tmp355;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp355;
tmp355[i0][i1][i2][i3] = __tmp_in_tmp355;
}
}
}
}

auto tmp356 = make_vector<int64_t>( (int32_t)768);
/* Variable to read the clear value corresponding to the input variable tmp356 at (1815,1-1815,38) */
uint64_t __tmp_in_tmp356;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
cin >> __tmp_in_tmp356;
tmp356[i0] = __tmp_in_tmp356;
}

auto tmp357 = make_vector<int64_t>( (int32_t)768);
/* Variable to read the clear value corresponding to the input variable tmp357 at (1817,1-1817,38) */
uint64_t __tmp_in_tmp357;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
cin >> __tmp_in_tmp357;
tmp357[i0] = __tmp_in_tmp357;
}

auto tmp358 = make_vector<int64_t>( (int32_t)768);
/* Variable to read the clear value corresponding to the input variable tmp358 at (1819,1-1819,38) */
uint64_t __tmp_in_tmp358;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
cin >> __tmp_in_tmp358;
tmp358[i0] = __tmp_in_tmp358;
}

auto tmp359 = make_vector<int64_t>( (int32_t)768);
/* Variable to read the clear value corresponding to the input variable tmp359 at (1821,1-1821,38) */
uint64_t __tmp_in_tmp359;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
cin >> __tmp_in_tmp359;
tmp359[i0] = __tmp_in_tmp359;
}

auto tmp360 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)768,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp360 at (1823,1-1823,49) */
uint64_t __tmp_in_tmp360;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)768; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp360;
tmp360[i0][i1][i2][i3] = __tmp_in_tmp360;
}
}
}
}

auto tmp361 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp361 at (1825,1-1825,38) */
uint64_t __tmp_in_tmp361;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp361;
tmp361[i0] = __tmp_in_tmp361;
}

auto tmp362 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp362 at (1827,1-1827,38) */
uint64_t __tmp_in_tmp362;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp362;
tmp362[i0] = __tmp_in_tmp362;
}

auto tmp363 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp363 at (1829,1-1829,38) */
uint64_t __tmp_in_tmp363;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp363;
tmp363[i0] = __tmp_in_tmp363;
}

auto tmp364 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp364 at (1831,1-1831,38) */
uint64_t __tmp_in_tmp364;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp364;
tmp364[i0] = __tmp_in_tmp364;
}

auto tmp365 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp365 at (1833,1-1833,48) */
uint64_t __tmp_in_tmp365;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp365;
tmp365[i0][i1][i2][i3] = __tmp_in_tmp365;
}
}
}
}

auto tmp366 = make_vector<int64_t>( (int32_t)800);
/* Variable to read the clear value corresponding to the input variable tmp366 at (1835,1-1835,38) */
uint64_t __tmp_in_tmp366;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
cin >> __tmp_in_tmp366;
tmp366[i0] = __tmp_in_tmp366;
}

auto tmp367 = make_vector<int64_t>( (int32_t)800);
/* Variable to read the clear value corresponding to the input variable tmp367 at (1837,1-1837,38) */
uint64_t __tmp_in_tmp367;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
cin >> __tmp_in_tmp367;
tmp367[i0] = __tmp_in_tmp367;
}

auto tmp368 = make_vector<int64_t>( (int32_t)800);
/* Variable to read the clear value corresponding to the input variable tmp368 at (1839,1-1839,38) */
uint64_t __tmp_in_tmp368;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
cin >> __tmp_in_tmp368;
tmp368[i0] = __tmp_in_tmp368;
}

auto tmp369 = make_vector<int64_t>( (int32_t)800);
/* Variable to read the clear value corresponding to the input variable tmp369 at (1841,1-1841,38) */
uint64_t __tmp_in_tmp369;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
cin >> __tmp_in_tmp369;
tmp369[i0] = __tmp_in_tmp369;
}

auto tmp370 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)800,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp370 at (1843,1-1843,49) */
uint64_t __tmp_in_tmp370;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)800; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp370;
tmp370[i0][i1][i2][i3] = __tmp_in_tmp370;
}
}
}
}

auto tmp371 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp371 at (1845,1-1845,38) */
uint64_t __tmp_in_tmp371;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp371;
tmp371[i0] = __tmp_in_tmp371;
}

auto tmp372 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp372 at (1847,1-1847,38) */
uint64_t __tmp_in_tmp372;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp372;
tmp372[i0] = __tmp_in_tmp372;
}

auto tmp373 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp373 at (1849,1-1849,38) */
uint64_t __tmp_in_tmp373;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp373;
tmp373[i0] = __tmp_in_tmp373;
}

auto tmp374 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp374 at (1851,1-1851,38) */
uint64_t __tmp_in_tmp374;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp374;
tmp374[i0] = __tmp_in_tmp374;
}

auto tmp375 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp375 at (1853,1-1853,48) */
uint64_t __tmp_in_tmp375;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp375;
tmp375[i0][i1][i2][i3] = __tmp_in_tmp375;
}
}
}
}

auto tmp376 = make_vector<int64_t>( (int32_t)832);
/* Variable to read the clear value corresponding to the input variable tmp376 at (1855,1-1855,38) */
uint64_t __tmp_in_tmp376;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
cin >> __tmp_in_tmp376;
tmp376[i0] = __tmp_in_tmp376;
}

auto tmp377 = make_vector<int64_t>( (int32_t)832);
/* Variable to read the clear value corresponding to the input variable tmp377 at (1857,1-1857,38) */
uint64_t __tmp_in_tmp377;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
cin >> __tmp_in_tmp377;
tmp377[i0] = __tmp_in_tmp377;
}

auto tmp378 = make_vector<int64_t>( (int32_t)832);
/* Variable to read the clear value corresponding to the input variable tmp378 at (1859,1-1859,38) */
uint64_t __tmp_in_tmp378;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
cin >> __tmp_in_tmp378;
tmp378[i0] = __tmp_in_tmp378;
}

auto tmp379 = make_vector<int64_t>( (int32_t)832);
/* Variable to read the clear value corresponding to the input variable tmp379 at (1861,1-1861,38) */
uint64_t __tmp_in_tmp379;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
cin >> __tmp_in_tmp379;
tmp379[i0] = __tmp_in_tmp379;
}

auto tmp380 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)832,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp380 at (1863,1-1863,49) */
uint64_t __tmp_in_tmp380;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)832; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp380;
tmp380[i0][i1][i2][i3] = __tmp_in_tmp380;
}
}
}
}

auto tmp381 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp381 at (1865,1-1865,38) */
uint64_t __tmp_in_tmp381;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp381;
tmp381[i0] = __tmp_in_tmp381;
}

auto tmp382 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp382 at (1867,1-1867,38) */
uint64_t __tmp_in_tmp382;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp382;
tmp382[i0] = __tmp_in_tmp382;
}

auto tmp383 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp383 at (1869,1-1869,38) */
uint64_t __tmp_in_tmp383;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp383;
tmp383[i0] = __tmp_in_tmp383;
}

auto tmp384 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp384 at (1871,1-1871,38) */
uint64_t __tmp_in_tmp384;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp384;
tmp384[i0] = __tmp_in_tmp384;
}

auto tmp385 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp385 at (1873,1-1873,48) */
uint64_t __tmp_in_tmp385;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp385;
tmp385[i0][i1][i2][i3] = __tmp_in_tmp385;
}
}
}
}

auto tmp386 = make_vector<int64_t>( (int32_t)864);
/* Variable to read the clear value corresponding to the input variable tmp386 at (1875,1-1875,38) */
uint64_t __tmp_in_tmp386;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
cin >> __tmp_in_tmp386;
tmp386[i0] = __tmp_in_tmp386;
}

auto tmp387 = make_vector<int64_t>( (int32_t)864);
/* Variable to read the clear value corresponding to the input variable tmp387 at (1877,1-1877,38) */
uint64_t __tmp_in_tmp387;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
cin >> __tmp_in_tmp387;
tmp387[i0] = __tmp_in_tmp387;
}

auto tmp388 = make_vector<int64_t>( (int32_t)864);
/* Variable to read the clear value corresponding to the input variable tmp388 at (1879,1-1879,38) */
uint64_t __tmp_in_tmp388;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
cin >> __tmp_in_tmp388;
tmp388[i0] = __tmp_in_tmp388;
}

auto tmp389 = make_vector<int64_t>( (int32_t)864);
/* Variable to read the clear value corresponding to the input variable tmp389 at (1881,1-1881,38) */
uint64_t __tmp_in_tmp389;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
cin >> __tmp_in_tmp389;
tmp389[i0] = __tmp_in_tmp389;
}

auto tmp390 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)864,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp390 at (1883,1-1883,49) */
uint64_t __tmp_in_tmp390;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)864; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp390;
tmp390[i0][i1][i2][i3] = __tmp_in_tmp390;
}
}
}
}

auto tmp391 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp391 at (1885,1-1885,38) */
uint64_t __tmp_in_tmp391;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp391;
tmp391[i0] = __tmp_in_tmp391;
}

auto tmp392 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp392 at (1887,1-1887,38) */
uint64_t __tmp_in_tmp392;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp392;
tmp392[i0] = __tmp_in_tmp392;
}

auto tmp393 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp393 at (1889,1-1889,38) */
uint64_t __tmp_in_tmp393;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp393;
tmp393[i0] = __tmp_in_tmp393;
}

auto tmp394 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp394 at (1891,1-1891,38) */
uint64_t __tmp_in_tmp394;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp394;
tmp394[i0] = __tmp_in_tmp394;
}

auto tmp395 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp395 at (1893,1-1893,48) */
uint64_t __tmp_in_tmp395;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp395;
tmp395[i0][i1][i2][i3] = __tmp_in_tmp395;
}
}
}
}

auto tmp396 = make_vector<int64_t>( (int32_t)896);
/* Variable to read the clear value corresponding to the input variable tmp396 at (1895,1-1895,38) */
uint64_t __tmp_in_tmp396;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
cin >> __tmp_in_tmp396;
tmp396[i0] = __tmp_in_tmp396;
}

auto tmp397 = make_vector<int64_t>( (int32_t)896);
/* Variable to read the clear value corresponding to the input variable tmp397 at (1897,1-1897,38) */
uint64_t __tmp_in_tmp397;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
cin >> __tmp_in_tmp397;
tmp397[i0] = __tmp_in_tmp397;
}

auto tmp398 = make_vector<int64_t>( (int32_t)896);
/* Variable to read the clear value corresponding to the input variable tmp398 at (1899,1-1899,38) */
uint64_t __tmp_in_tmp398;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
cin >> __tmp_in_tmp398;
tmp398[i0] = __tmp_in_tmp398;
}

auto tmp399 = make_vector<int64_t>( (int32_t)896);
/* Variable to read the clear value corresponding to the input variable tmp399 at (1901,1-1901,38) */
uint64_t __tmp_in_tmp399;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
cin >> __tmp_in_tmp399;
tmp399[i0] = __tmp_in_tmp399;
}

auto tmp400 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)896,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp400 at (1903,1-1903,49) */
uint64_t __tmp_in_tmp400;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)896; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp400;
tmp400[i0][i1][i2][i3] = __tmp_in_tmp400;
}
}
}
}

auto tmp401 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp401 at (1905,1-1905,38) */
uint64_t __tmp_in_tmp401;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp401;
tmp401[i0] = __tmp_in_tmp401;
}

auto tmp402 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp402 at (1907,1-1907,38) */
uint64_t __tmp_in_tmp402;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp402;
tmp402[i0] = __tmp_in_tmp402;
}

auto tmp403 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp403 at (1909,1-1909,38) */
uint64_t __tmp_in_tmp403;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp403;
tmp403[i0] = __tmp_in_tmp403;
}

auto tmp404 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp404 at (1911,1-1911,38) */
uint64_t __tmp_in_tmp404;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp404;
tmp404[i0] = __tmp_in_tmp404;
}

auto tmp405 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp405 at (1913,1-1913,48) */
uint64_t __tmp_in_tmp405;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp405;
tmp405[i0][i1][i2][i3] = __tmp_in_tmp405;
}
}
}
}

auto tmp406 = make_vector<int64_t>( (int32_t)928);
/* Variable to read the clear value corresponding to the input variable tmp406 at (1915,1-1915,38) */
uint64_t __tmp_in_tmp406;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
cin >> __tmp_in_tmp406;
tmp406[i0] = __tmp_in_tmp406;
}

auto tmp407 = make_vector<int64_t>( (int32_t)928);
/* Variable to read the clear value corresponding to the input variable tmp407 at (1917,1-1917,38) */
uint64_t __tmp_in_tmp407;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
cin >> __tmp_in_tmp407;
tmp407[i0] = __tmp_in_tmp407;
}

auto tmp408 = make_vector<int64_t>( (int32_t)928);
/* Variable to read the clear value corresponding to the input variable tmp408 at (1919,1-1919,38) */
uint64_t __tmp_in_tmp408;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
cin >> __tmp_in_tmp408;
tmp408[i0] = __tmp_in_tmp408;
}

auto tmp409 = make_vector<int64_t>( (int32_t)928);
/* Variable to read the clear value corresponding to the input variable tmp409 at (1921,1-1921,38) */
uint64_t __tmp_in_tmp409;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
cin >> __tmp_in_tmp409;
tmp409[i0] = __tmp_in_tmp409;
}

auto tmp410 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)928,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp410 at (1923,1-1923,49) */
uint64_t __tmp_in_tmp410;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)928; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp410;
tmp410[i0][i1][i2][i3] = __tmp_in_tmp410;
}
}
}
}

auto tmp411 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp411 at (1925,1-1925,38) */
uint64_t __tmp_in_tmp411;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp411;
tmp411[i0] = __tmp_in_tmp411;
}

auto tmp412 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp412 at (1927,1-1927,38) */
uint64_t __tmp_in_tmp412;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp412;
tmp412[i0] = __tmp_in_tmp412;
}

auto tmp413 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp413 at (1929,1-1929,38) */
uint64_t __tmp_in_tmp413;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp413;
tmp413[i0] = __tmp_in_tmp413;
}

auto tmp414 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp414 at (1931,1-1931,38) */
uint64_t __tmp_in_tmp414;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp414;
tmp414[i0] = __tmp_in_tmp414;
}

auto tmp415 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp415 at (1933,1-1933,48) */
uint64_t __tmp_in_tmp415;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp415;
tmp415[i0][i1][i2][i3] = __tmp_in_tmp415;
}
}
}
}

auto tmp416 = make_vector<int64_t>( (int32_t)960);
/* Variable to read the clear value corresponding to the input variable tmp416 at (1935,1-1935,38) */
uint64_t __tmp_in_tmp416;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
cin >> __tmp_in_tmp416;
tmp416[i0] = __tmp_in_tmp416;
}

auto tmp417 = make_vector<int64_t>( (int32_t)960);
/* Variable to read the clear value corresponding to the input variable tmp417 at (1937,1-1937,38) */
uint64_t __tmp_in_tmp417;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
cin >> __tmp_in_tmp417;
tmp417[i0] = __tmp_in_tmp417;
}

auto tmp418 = make_vector<int64_t>( (int32_t)960);
/* Variable to read the clear value corresponding to the input variable tmp418 at (1939,1-1939,38) */
uint64_t __tmp_in_tmp418;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
cin >> __tmp_in_tmp418;
tmp418[i0] = __tmp_in_tmp418;
}

auto tmp419 = make_vector<int64_t>( (int32_t)960);
/* Variable to read the clear value corresponding to the input variable tmp419 at (1941,1-1941,38) */
uint64_t __tmp_in_tmp419;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
cin >> __tmp_in_tmp419;
tmp419[i0] = __tmp_in_tmp419;
}

auto tmp420 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)960,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp420 at (1943,1-1943,49) */
uint64_t __tmp_in_tmp420;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)960; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp420;
tmp420[i0][i1][i2][i3] = __tmp_in_tmp420;
}
}
}
}

auto tmp421 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp421 at (1945,1-1945,38) */
uint64_t __tmp_in_tmp421;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp421;
tmp421[i0] = __tmp_in_tmp421;
}

auto tmp422 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp422 at (1947,1-1947,38) */
uint64_t __tmp_in_tmp422;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp422;
tmp422[i0] = __tmp_in_tmp422;
}

auto tmp423 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp423 at (1949,1-1949,38) */
uint64_t __tmp_in_tmp423;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp423;
tmp423[i0] = __tmp_in_tmp423;
}

auto tmp424 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp424 at (1951,1-1951,38) */
uint64_t __tmp_in_tmp424;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp424;
tmp424[i0] = __tmp_in_tmp424;
}

auto tmp425 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp425 at (1953,1-1953,48) */
uint64_t __tmp_in_tmp425;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp425;
tmp425[i0][i1][i2][i3] = __tmp_in_tmp425;
}
}
}
}

auto tmp426 = make_vector<int64_t>( (int32_t)992);
/* Variable to read the clear value corresponding to the input variable tmp426 at (1955,1-1955,38) */
uint64_t __tmp_in_tmp426;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
cin >> __tmp_in_tmp426;
tmp426[i0] = __tmp_in_tmp426;
}

auto tmp427 = make_vector<int64_t>( (int32_t)992);
/* Variable to read the clear value corresponding to the input variable tmp427 at (1957,1-1957,38) */
uint64_t __tmp_in_tmp427;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
cin >> __tmp_in_tmp427;
tmp427[i0] = __tmp_in_tmp427;
}

auto tmp428 = make_vector<int64_t>( (int32_t)992);
/* Variable to read the clear value corresponding to the input variable tmp428 at (1959,1-1959,38) */
uint64_t __tmp_in_tmp428;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
cin >> __tmp_in_tmp428;
tmp428[i0] = __tmp_in_tmp428;
}

auto tmp429 = make_vector<int64_t>( (int32_t)992);
/* Variable to read the clear value corresponding to the input variable tmp429 at (1961,1-1961,38) */
uint64_t __tmp_in_tmp429;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
cin >> __tmp_in_tmp429;
tmp429[i0] = __tmp_in_tmp429;
}

auto tmp430 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)992,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp430 at (1963,1-1963,49) */
uint64_t __tmp_in_tmp430;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)992; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp430;
tmp430[i0][i1][i2][i3] = __tmp_in_tmp430;
}
}
}
}

auto tmp431 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp431 at (1965,1-1965,38) */
uint64_t __tmp_in_tmp431;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp431;
tmp431[i0] = __tmp_in_tmp431;
}

auto tmp432 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp432 at (1967,1-1967,38) */
uint64_t __tmp_in_tmp432;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp432;
tmp432[i0] = __tmp_in_tmp432;
}

auto tmp433 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp433 at (1969,1-1969,38) */
uint64_t __tmp_in_tmp433;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp433;
tmp433[i0] = __tmp_in_tmp433;
}

auto tmp434 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp434 at (1971,1-1971,38) */
uint64_t __tmp_in_tmp434;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp434;
tmp434[i0] = __tmp_in_tmp434;
}

auto tmp435 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp435 at (1973,1-1973,48) */
uint64_t __tmp_in_tmp435;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp435;
tmp435[i0][i1][i2][i3] = __tmp_in_tmp435;
}
}
}
}

auto tmp436 = make_vector<int64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp436 at (1975,1-1975,39) */
uint64_t __tmp_in_tmp436;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp436;
tmp436[i0] = __tmp_in_tmp436;
}

auto tmp437 = make_vector<int64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp437 at (1977,1-1977,39) */
uint64_t __tmp_in_tmp437;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp437;
tmp437[i0] = __tmp_in_tmp437;
}

auto tmp438 = make_vector<int64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp438 at (1979,1-1979,39) */
uint64_t __tmp_in_tmp438;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp438;
tmp438[i0] = __tmp_in_tmp438;
}

auto tmp439 = make_vector<int64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp439 at (1981,1-1981,39) */
uint64_t __tmp_in_tmp439;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp439;
tmp439[i0] = __tmp_in_tmp439;
}

auto tmp440 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp440 at (1983,1-1983,50) */
uint64_t __tmp_in_tmp440;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
cin >> __tmp_in_tmp440;
tmp440[i0][i1][i2][i3] = __tmp_in_tmp440;
}
}
}
}

auto tmp441 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp441 at (1985,1-1985,38) */
uint64_t __tmp_in_tmp441;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp441;
tmp441[i0] = __tmp_in_tmp441;
}

auto tmp442 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp442 at (1987,1-1987,38) */
uint64_t __tmp_in_tmp442;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp442;
tmp442[i0] = __tmp_in_tmp442;
}

auto tmp443 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp443 at (1989,1-1989,38) */
uint64_t __tmp_in_tmp443;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp443;
tmp443[i0] = __tmp_in_tmp443;
}

auto tmp444 = make_vector<int64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp444 at (1991,1-1991,38) */
uint64_t __tmp_in_tmp444;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cin >> __tmp_in_tmp444;
tmp444[i0] = __tmp_in_tmp444;
}

auto tmp445 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp445 at (1993,1-1993,49) */
uint64_t __tmp_in_tmp445;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp445;
tmp445[i0][i1][i2][i3] = __tmp_in_tmp445;
}
}
}
}

auto tmp446 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp446 at (1995,1-1995,38) */
uint64_t __tmp_in_tmp446;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp446;
tmp446[i0] = __tmp_in_tmp446;
}

auto tmp447 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp447 at (1997,1-1997,38) */
uint64_t __tmp_in_tmp447;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp447;
tmp447[i0] = __tmp_in_tmp447;
}

auto tmp448 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp448 at (1999,1-1999,38) */
uint64_t __tmp_in_tmp448;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp448;
tmp448[i0] = __tmp_in_tmp448;
}

auto tmp449 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp449 at (2001,1-2001,38) */
uint64_t __tmp_in_tmp449;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp449;
tmp449[i0] = __tmp_in_tmp449;
}

auto tmp450 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp450 at (2003,1-2003,48) */
uint64_t __tmp_in_tmp450;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp450;
tmp450[i0][i1][i2][i3] = __tmp_in_tmp450;
}
}
}
}

auto tmp451 = make_vector<int64_t>( (int32_t)544);
/* Variable to read the clear value corresponding to the input variable tmp451 at (2005,1-2005,38) */
uint64_t __tmp_in_tmp451;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
cin >> __tmp_in_tmp451;
tmp451[i0] = __tmp_in_tmp451;
}

auto tmp452 = make_vector<int64_t>( (int32_t)544);
/* Variable to read the clear value corresponding to the input variable tmp452 at (2007,1-2007,38) */
uint64_t __tmp_in_tmp452;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
cin >> __tmp_in_tmp452;
tmp452[i0] = __tmp_in_tmp452;
}

auto tmp453 = make_vector<int64_t>( (int32_t)544);
/* Variable to read the clear value corresponding to the input variable tmp453 at (2009,1-2009,38) */
uint64_t __tmp_in_tmp453;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
cin >> __tmp_in_tmp453;
tmp453[i0] = __tmp_in_tmp453;
}

auto tmp454 = make_vector<int64_t>( (int32_t)544);
/* Variable to read the clear value corresponding to the input variable tmp454 at (2011,1-2011,38) */
uint64_t __tmp_in_tmp454;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
cin >> __tmp_in_tmp454;
tmp454[i0] = __tmp_in_tmp454;
}

auto tmp455 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)544,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp455 at (2013,1-2013,49) */
uint64_t __tmp_in_tmp455;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)544; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp455;
tmp455[i0][i1][i2][i3] = __tmp_in_tmp455;
}
}
}
}

auto tmp456 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp456 at (2015,1-2015,38) */
uint64_t __tmp_in_tmp456;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp456;
tmp456[i0] = __tmp_in_tmp456;
}

auto tmp457 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp457 at (2017,1-2017,38) */
uint64_t __tmp_in_tmp457;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp457;
tmp457[i0] = __tmp_in_tmp457;
}

auto tmp458 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp458 at (2019,1-2019,38) */
uint64_t __tmp_in_tmp458;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp458;
tmp458[i0] = __tmp_in_tmp458;
}

auto tmp459 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp459 at (2021,1-2021,38) */
uint64_t __tmp_in_tmp459;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp459;
tmp459[i0] = __tmp_in_tmp459;
}

auto tmp460 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp460 at (2023,1-2023,48) */
uint64_t __tmp_in_tmp460;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp460;
tmp460[i0][i1][i2][i3] = __tmp_in_tmp460;
}
}
}
}

auto tmp461 = make_vector<int64_t>( (int32_t)576);
/* Variable to read the clear value corresponding to the input variable tmp461 at (2025,1-2025,38) */
uint64_t __tmp_in_tmp461;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
cin >> __tmp_in_tmp461;
tmp461[i0] = __tmp_in_tmp461;
}

auto tmp462 = make_vector<int64_t>( (int32_t)576);
/* Variable to read the clear value corresponding to the input variable tmp462 at (2027,1-2027,38) */
uint64_t __tmp_in_tmp462;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
cin >> __tmp_in_tmp462;
tmp462[i0] = __tmp_in_tmp462;
}

auto tmp463 = make_vector<int64_t>( (int32_t)576);
/* Variable to read the clear value corresponding to the input variable tmp463 at (2029,1-2029,38) */
uint64_t __tmp_in_tmp463;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
cin >> __tmp_in_tmp463;
tmp463[i0] = __tmp_in_tmp463;
}

auto tmp464 = make_vector<int64_t>( (int32_t)576);
/* Variable to read the clear value corresponding to the input variable tmp464 at (2031,1-2031,38) */
uint64_t __tmp_in_tmp464;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
cin >> __tmp_in_tmp464;
tmp464[i0] = __tmp_in_tmp464;
}

auto tmp465 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)576,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp465 at (2033,1-2033,49) */
uint64_t __tmp_in_tmp465;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)576; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp465;
tmp465[i0][i1][i2][i3] = __tmp_in_tmp465;
}
}
}
}

auto tmp466 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp466 at (2035,1-2035,38) */
uint64_t __tmp_in_tmp466;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp466;
tmp466[i0] = __tmp_in_tmp466;
}

auto tmp467 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp467 at (2037,1-2037,38) */
uint64_t __tmp_in_tmp467;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp467;
tmp467[i0] = __tmp_in_tmp467;
}

auto tmp468 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp468 at (2039,1-2039,38) */
uint64_t __tmp_in_tmp468;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp468;
tmp468[i0] = __tmp_in_tmp468;
}

auto tmp469 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp469 at (2041,1-2041,38) */
uint64_t __tmp_in_tmp469;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp469;
tmp469[i0] = __tmp_in_tmp469;
}

auto tmp470 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp470 at (2043,1-2043,48) */
uint64_t __tmp_in_tmp470;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp470;
tmp470[i0][i1][i2][i3] = __tmp_in_tmp470;
}
}
}
}

auto tmp471 = make_vector<int64_t>( (int32_t)608);
/* Variable to read the clear value corresponding to the input variable tmp471 at (2045,1-2045,38) */
uint64_t __tmp_in_tmp471;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
cin >> __tmp_in_tmp471;
tmp471[i0] = __tmp_in_tmp471;
}

auto tmp472 = make_vector<int64_t>( (int32_t)608);
/* Variable to read the clear value corresponding to the input variable tmp472 at (2047,1-2047,38) */
uint64_t __tmp_in_tmp472;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
cin >> __tmp_in_tmp472;
tmp472[i0] = __tmp_in_tmp472;
}

auto tmp473 = make_vector<int64_t>( (int32_t)608);
/* Variable to read the clear value corresponding to the input variable tmp473 at (2049,1-2049,38) */
uint64_t __tmp_in_tmp473;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
cin >> __tmp_in_tmp473;
tmp473[i0] = __tmp_in_tmp473;
}

auto tmp474 = make_vector<int64_t>( (int32_t)608);
/* Variable to read the clear value corresponding to the input variable tmp474 at (2051,1-2051,38) */
uint64_t __tmp_in_tmp474;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
cin >> __tmp_in_tmp474;
tmp474[i0] = __tmp_in_tmp474;
}

auto tmp475 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)608,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp475 at (2053,1-2053,49) */
uint64_t __tmp_in_tmp475;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)608; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp475;
tmp475[i0][i1][i2][i3] = __tmp_in_tmp475;
}
}
}
}

auto tmp476 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp476 at (2055,1-2055,38) */
uint64_t __tmp_in_tmp476;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp476;
tmp476[i0] = __tmp_in_tmp476;
}

auto tmp477 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp477 at (2057,1-2057,38) */
uint64_t __tmp_in_tmp477;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp477;
tmp477[i0] = __tmp_in_tmp477;
}

auto tmp478 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp478 at (2059,1-2059,38) */
uint64_t __tmp_in_tmp478;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp478;
tmp478[i0] = __tmp_in_tmp478;
}

auto tmp479 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp479 at (2061,1-2061,38) */
uint64_t __tmp_in_tmp479;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp479;
tmp479[i0] = __tmp_in_tmp479;
}

auto tmp480 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp480 at (2063,1-2063,48) */
uint64_t __tmp_in_tmp480;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp480;
tmp480[i0][i1][i2][i3] = __tmp_in_tmp480;
}
}
}
}

auto tmp481 = make_vector<int64_t>( (int32_t)640);
/* Variable to read the clear value corresponding to the input variable tmp481 at (2065,1-2065,38) */
uint64_t __tmp_in_tmp481;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
cin >> __tmp_in_tmp481;
tmp481[i0] = __tmp_in_tmp481;
}

auto tmp482 = make_vector<int64_t>( (int32_t)640);
/* Variable to read the clear value corresponding to the input variable tmp482 at (2067,1-2067,38) */
uint64_t __tmp_in_tmp482;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
cin >> __tmp_in_tmp482;
tmp482[i0] = __tmp_in_tmp482;
}

auto tmp483 = make_vector<int64_t>( (int32_t)640);
/* Variable to read the clear value corresponding to the input variable tmp483 at (2069,1-2069,38) */
uint64_t __tmp_in_tmp483;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
cin >> __tmp_in_tmp483;
tmp483[i0] = __tmp_in_tmp483;
}

auto tmp484 = make_vector<int64_t>( (int32_t)640);
/* Variable to read the clear value corresponding to the input variable tmp484 at (2071,1-2071,38) */
uint64_t __tmp_in_tmp484;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
cin >> __tmp_in_tmp484;
tmp484[i0] = __tmp_in_tmp484;
}

auto tmp485 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)640,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp485 at (2073,1-2073,49) */
uint64_t __tmp_in_tmp485;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)640; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp485;
tmp485[i0][i1][i2][i3] = __tmp_in_tmp485;
}
}
}
}

auto tmp486 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp486 at (2075,1-2075,38) */
uint64_t __tmp_in_tmp486;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp486;
tmp486[i0] = __tmp_in_tmp486;
}

auto tmp487 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp487 at (2077,1-2077,38) */
uint64_t __tmp_in_tmp487;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp487;
tmp487[i0] = __tmp_in_tmp487;
}

auto tmp488 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp488 at (2079,1-2079,38) */
uint64_t __tmp_in_tmp488;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp488;
tmp488[i0] = __tmp_in_tmp488;
}

auto tmp489 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp489 at (2081,1-2081,38) */
uint64_t __tmp_in_tmp489;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp489;
tmp489[i0] = __tmp_in_tmp489;
}

auto tmp490 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp490 at (2083,1-2083,48) */
uint64_t __tmp_in_tmp490;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp490;
tmp490[i0][i1][i2][i3] = __tmp_in_tmp490;
}
}
}
}

auto tmp491 = make_vector<int64_t>( (int32_t)672);
/* Variable to read the clear value corresponding to the input variable tmp491 at (2085,1-2085,38) */
uint64_t __tmp_in_tmp491;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
cin >> __tmp_in_tmp491;
tmp491[i0] = __tmp_in_tmp491;
}

auto tmp492 = make_vector<int64_t>( (int32_t)672);
/* Variable to read the clear value corresponding to the input variable tmp492 at (2087,1-2087,38) */
uint64_t __tmp_in_tmp492;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
cin >> __tmp_in_tmp492;
tmp492[i0] = __tmp_in_tmp492;
}

auto tmp493 = make_vector<int64_t>( (int32_t)672);
/* Variable to read the clear value corresponding to the input variable tmp493 at (2089,1-2089,38) */
uint64_t __tmp_in_tmp493;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
cin >> __tmp_in_tmp493;
tmp493[i0] = __tmp_in_tmp493;
}

auto tmp494 = make_vector<int64_t>( (int32_t)672);
/* Variable to read the clear value corresponding to the input variable tmp494 at (2091,1-2091,38) */
uint64_t __tmp_in_tmp494;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
cin >> __tmp_in_tmp494;
tmp494[i0] = __tmp_in_tmp494;
}

auto tmp495 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)672,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp495 at (2093,1-2093,49) */
uint64_t __tmp_in_tmp495;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)672; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp495;
tmp495[i0][i1][i2][i3] = __tmp_in_tmp495;
}
}
}
}

auto tmp496 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp496 at (2095,1-2095,38) */
uint64_t __tmp_in_tmp496;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp496;
tmp496[i0] = __tmp_in_tmp496;
}

auto tmp497 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp497 at (2097,1-2097,38) */
uint64_t __tmp_in_tmp497;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp497;
tmp497[i0] = __tmp_in_tmp497;
}

auto tmp498 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp498 at (2099,1-2099,38) */
uint64_t __tmp_in_tmp498;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp498;
tmp498[i0] = __tmp_in_tmp498;
}

auto tmp499 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp499 at (2101,1-2101,38) */
uint64_t __tmp_in_tmp499;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp499;
tmp499[i0] = __tmp_in_tmp499;
}

auto tmp500 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp500 at (2103,1-2103,48) */
uint64_t __tmp_in_tmp500;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp500;
tmp500[i0][i1][i2][i3] = __tmp_in_tmp500;
}
}
}
}

auto tmp501 = make_vector<int64_t>( (int32_t)704);
/* Variable to read the clear value corresponding to the input variable tmp501 at (2105,1-2105,38) */
uint64_t __tmp_in_tmp501;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
cin >> __tmp_in_tmp501;
tmp501[i0] = __tmp_in_tmp501;
}

auto tmp502 = make_vector<int64_t>( (int32_t)704);
/* Variable to read the clear value corresponding to the input variable tmp502 at (2107,1-2107,38) */
uint64_t __tmp_in_tmp502;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
cin >> __tmp_in_tmp502;
tmp502[i0] = __tmp_in_tmp502;
}

auto tmp503 = make_vector<int64_t>( (int32_t)704);
/* Variable to read the clear value corresponding to the input variable tmp503 at (2109,1-2109,38) */
uint64_t __tmp_in_tmp503;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
cin >> __tmp_in_tmp503;
tmp503[i0] = __tmp_in_tmp503;
}

auto tmp504 = make_vector<int64_t>( (int32_t)704);
/* Variable to read the clear value corresponding to the input variable tmp504 at (2111,1-2111,38) */
uint64_t __tmp_in_tmp504;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
cin >> __tmp_in_tmp504;
tmp504[i0] = __tmp_in_tmp504;
}

auto tmp505 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)704,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp505 at (2113,1-2113,49) */
uint64_t __tmp_in_tmp505;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)704; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp505;
tmp505[i0][i1][i2][i3] = __tmp_in_tmp505;
}
}
}
}

auto tmp506 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp506 at (2115,1-2115,38) */
uint64_t __tmp_in_tmp506;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp506;
tmp506[i0] = __tmp_in_tmp506;
}

auto tmp507 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp507 at (2117,1-2117,38) */
uint64_t __tmp_in_tmp507;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp507;
tmp507[i0] = __tmp_in_tmp507;
}

auto tmp508 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp508 at (2119,1-2119,38) */
uint64_t __tmp_in_tmp508;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp508;
tmp508[i0] = __tmp_in_tmp508;
}

auto tmp509 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp509 at (2121,1-2121,38) */
uint64_t __tmp_in_tmp509;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp509;
tmp509[i0] = __tmp_in_tmp509;
}

auto tmp510 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp510 at (2123,1-2123,48) */
uint64_t __tmp_in_tmp510;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp510;
tmp510[i0][i1][i2][i3] = __tmp_in_tmp510;
}
}
}
}

auto tmp511 = make_vector<int64_t>( (int32_t)736);
/* Variable to read the clear value corresponding to the input variable tmp511 at (2125,1-2125,38) */
uint64_t __tmp_in_tmp511;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
cin >> __tmp_in_tmp511;
tmp511[i0] = __tmp_in_tmp511;
}

auto tmp512 = make_vector<int64_t>( (int32_t)736);
/* Variable to read the clear value corresponding to the input variable tmp512 at (2127,1-2127,38) */
uint64_t __tmp_in_tmp512;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
cin >> __tmp_in_tmp512;
tmp512[i0] = __tmp_in_tmp512;
}

auto tmp513 = make_vector<int64_t>( (int32_t)736);
/* Variable to read the clear value corresponding to the input variable tmp513 at (2129,1-2129,38) */
uint64_t __tmp_in_tmp513;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
cin >> __tmp_in_tmp513;
tmp513[i0] = __tmp_in_tmp513;
}

auto tmp514 = make_vector<int64_t>( (int32_t)736);
/* Variable to read the clear value corresponding to the input variable tmp514 at (2131,1-2131,38) */
uint64_t __tmp_in_tmp514;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
cin >> __tmp_in_tmp514;
tmp514[i0] = __tmp_in_tmp514;
}

auto tmp515 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)736,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp515 at (2133,1-2133,49) */
uint64_t __tmp_in_tmp515;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)736; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp515;
tmp515[i0][i1][i2][i3] = __tmp_in_tmp515;
}
}
}
}

auto tmp516 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp516 at (2135,1-2135,38) */
uint64_t __tmp_in_tmp516;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp516;
tmp516[i0] = __tmp_in_tmp516;
}

auto tmp517 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp517 at (2137,1-2137,38) */
uint64_t __tmp_in_tmp517;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp517;
tmp517[i0] = __tmp_in_tmp517;
}

auto tmp518 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp518 at (2139,1-2139,38) */
uint64_t __tmp_in_tmp518;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp518;
tmp518[i0] = __tmp_in_tmp518;
}

auto tmp519 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp519 at (2141,1-2141,38) */
uint64_t __tmp_in_tmp519;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp519;
tmp519[i0] = __tmp_in_tmp519;
}

auto tmp520 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp520 at (2143,1-2143,48) */
uint64_t __tmp_in_tmp520;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp520;
tmp520[i0][i1][i2][i3] = __tmp_in_tmp520;
}
}
}
}

auto tmp521 = make_vector<int64_t>( (int32_t)768);
/* Variable to read the clear value corresponding to the input variable tmp521 at (2145,1-2145,38) */
uint64_t __tmp_in_tmp521;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
cin >> __tmp_in_tmp521;
tmp521[i0] = __tmp_in_tmp521;
}

auto tmp522 = make_vector<int64_t>( (int32_t)768);
/* Variable to read the clear value corresponding to the input variable tmp522 at (2147,1-2147,38) */
uint64_t __tmp_in_tmp522;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
cin >> __tmp_in_tmp522;
tmp522[i0] = __tmp_in_tmp522;
}

auto tmp523 = make_vector<int64_t>( (int32_t)768);
/* Variable to read the clear value corresponding to the input variable tmp523 at (2149,1-2149,38) */
uint64_t __tmp_in_tmp523;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
cin >> __tmp_in_tmp523;
tmp523[i0] = __tmp_in_tmp523;
}

auto tmp524 = make_vector<int64_t>( (int32_t)768);
/* Variable to read the clear value corresponding to the input variable tmp524 at (2151,1-2151,38) */
uint64_t __tmp_in_tmp524;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
cin >> __tmp_in_tmp524;
tmp524[i0] = __tmp_in_tmp524;
}

auto tmp525 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)768,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp525 at (2153,1-2153,49) */
uint64_t __tmp_in_tmp525;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)768; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp525;
tmp525[i0][i1][i2][i3] = __tmp_in_tmp525;
}
}
}
}

auto tmp526 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp526 at (2155,1-2155,38) */
uint64_t __tmp_in_tmp526;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp526;
tmp526[i0] = __tmp_in_tmp526;
}

auto tmp527 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp527 at (2157,1-2157,38) */
uint64_t __tmp_in_tmp527;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp527;
tmp527[i0] = __tmp_in_tmp527;
}

auto tmp528 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp528 at (2159,1-2159,38) */
uint64_t __tmp_in_tmp528;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp528;
tmp528[i0] = __tmp_in_tmp528;
}

auto tmp529 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp529 at (2161,1-2161,38) */
uint64_t __tmp_in_tmp529;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp529;
tmp529[i0] = __tmp_in_tmp529;
}

auto tmp530 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp530 at (2163,1-2163,48) */
uint64_t __tmp_in_tmp530;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp530;
tmp530[i0][i1][i2][i3] = __tmp_in_tmp530;
}
}
}
}

auto tmp531 = make_vector<int64_t>( (int32_t)800);
/* Variable to read the clear value corresponding to the input variable tmp531 at (2165,1-2165,38) */
uint64_t __tmp_in_tmp531;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
cin >> __tmp_in_tmp531;
tmp531[i0] = __tmp_in_tmp531;
}

auto tmp532 = make_vector<int64_t>( (int32_t)800);
/* Variable to read the clear value corresponding to the input variable tmp532 at (2167,1-2167,38) */
uint64_t __tmp_in_tmp532;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
cin >> __tmp_in_tmp532;
tmp532[i0] = __tmp_in_tmp532;
}

auto tmp533 = make_vector<int64_t>( (int32_t)800);
/* Variable to read the clear value corresponding to the input variable tmp533 at (2169,1-2169,38) */
uint64_t __tmp_in_tmp533;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
cin >> __tmp_in_tmp533;
tmp533[i0] = __tmp_in_tmp533;
}

auto tmp534 = make_vector<int64_t>( (int32_t)800);
/* Variable to read the clear value corresponding to the input variable tmp534 at (2171,1-2171,38) */
uint64_t __tmp_in_tmp534;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
cin >> __tmp_in_tmp534;
tmp534[i0] = __tmp_in_tmp534;
}

auto tmp535 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)800,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp535 at (2173,1-2173,49) */
uint64_t __tmp_in_tmp535;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)800; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp535;
tmp535[i0][i1][i2][i3] = __tmp_in_tmp535;
}
}
}
}

auto tmp536 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp536 at (2175,1-2175,38) */
uint64_t __tmp_in_tmp536;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp536;
tmp536[i0] = __tmp_in_tmp536;
}

auto tmp537 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp537 at (2177,1-2177,38) */
uint64_t __tmp_in_tmp537;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp537;
tmp537[i0] = __tmp_in_tmp537;
}

auto tmp538 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp538 at (2179,1-2179,38) */
uint64_t __tmp_in_tmp538;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp538;
tmp538[i0] = __tmp_in_tmp538;
}

auto tmp539 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp539 at (2181,1-2181,38) */
uint64_t __tmp_in_tmp539;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp539;
tmp539[i0] = __tmp_in_tmp539;
}

auto tmp540 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp540 at (2183,1-2183,48) */
uint64_t __tmp_in_tmp540;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp540;
tmp540[i0][i1][i2][i3] = __tmp_in_tmp540;
}
}
}
}

auto tmp541 = make_vector<int64_t>( (int32_t)832);
/* Variable to read the clear value corresponding to the input variable tmp541 at (2185,1-2185,38) */
uint64_t __tmp_in_tmp541;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
cin >> __tmp_in_tmp541;
tmp541[i0] = __tmp_in_tmp541;
}

auto tmp542 = make_vector<int64_t>( (int32_t)832);
/* Variable to read the clear value corresponding to the input variable tmp542 at (2187,1-2187,38) */
uint64_t __tmp_in_tmp542;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
cin >> __tmp_in_tmp542;
tmp542[i0] = __tmp_in_tmp542;
}

auto tmp543 = make_vector<int64_t>( (int32_t)832);
/* Variable to read the clear value corresponding to the input variable tmp543 at (2189,1-2189,38) */
uint64_t __tmp_in_tmp543;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
cin >> __tmp_in_tmp543;
tmp543[i0] = __tmp_in_tmp543;
}

auto tmp544 = make_vector<int64_t>( (int32_t)832);
/* Variable to read the clear value corresponding to the input variable tmp544 at (2191,1-2191,38) */
uint64_t __tmp_in_tmp544;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
cin >> __tmp_in_tmp544;
tmp544[i0] = __tmp_in_tmp544;
}

auto tmp545 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)832,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp545 at (2193,1-2193,49) */
uint64_t __tmp_in_tmp545;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)832; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp545;
tmp545[i0][i1][i2][i3] = __tmp_in_tmp545;
}
}
}
}

auto tmp546 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp546 at (2195,1-2195,38) */
uint64_t __tmp_in_tmp546;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp546;
tmp546[i0] = __tmp_in_tmp546;
}

auto tmp547 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp547 at (2197,1-2197,38) */
uint64_t __tmp_in_tmp547;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp547;
tmp547[i0] = __tmp_in_tmp547;
}

auto tmp548 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp548 at (2199,1-2199,38) */
uint64_t __tmp_in_tmp548;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp548;
tmp548[i0] = __tmp_in_tmp548;
}

auto tmp549 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp549 at (2201,1-2201,38) */
uint64_t __tmp_in_tmp549;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp549;
tmp549[i0] = __tmp_in_tmp549;
}

auto tmp550 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp550 at (2203,1-2203,48) */
uint64_t __tmp_in_tmp550;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp550;
tmp550[i0][i1][i2][i3] = __tmp_in_tmp550;
}
}
}
}

auto tmp551 = make_vector<int64_t>( (int32_t)864);
/* Variable to read the clear value corresponding to the input variable tmp551 at (2205,1-2205,38) */
uint64_t __tmp_in_tmp551;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
cin >> __tmp_in_tmp551;
tmp551[i0] = __tmp_in_tmp551;
}

auto tmp552 = make_vector<int64_t>( (int32_t)864);
/* Variable to read the clear value corresponding to the input variable tmp552 at (2207,1-2207,38) */
uint64_t __tmp_in_tmp552;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
cin >> __tmp_in_tmp552;
tmp552[i0] = __tmp_in_tmp552;
}

auto tmp553 = make_vector<int64_t>( (int32_t)864);
/* Variable to read the clear value corresponding to the input variable tmp553 at (2209,1-2209,38) */
uint64_t __tmp_in_tmp553;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
cin >> __tmp_in_tmp553;
tmp553[i0] = __tmp_in_tmp553;
}

auto tmp554 = make_vector<int64_t>( (int32_t)864);
/* Variable to read the clear value corresponding to the input variable tmp554 at (2211,1-2211,38) */
uint64_t __tmp_in_tmp554;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
cin >> __tmp_in_tmp554;
tmp554[i0] = __tmp_in_tmp554;
}

auto tmp555 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)864,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp555 at (2213,1-2213,49) */
uint64_t __tmp_in_tmp555;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)864; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp555;
tmp555[i0][i1][i2][i3] = __tmp_in_tmp555;
}
}
}
}

auto tmp556 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp556 at (2215,1-2215,38) */
uint64_t __tmp_in_tmp556;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp556;
tmp556[i0] = __tmp_in_tmp556;
}

auto tmp557 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp557 at (2217,1-2217,38) */
uint64_t __tmp_in_tmp557;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp557;
tmp557[i0] = __tmp_in_tmp557;
}

auto tmp558 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp558 at (2219,1-2219,38) */
uint64_t __tmp_in_tmp558;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp558;
tmp558[i0] = __tmp_in_tmp558;
}

auto tmp559 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp559 at (2221,1-2221,38) */
uint64_t __tmp_in_tmp559;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp559;
tmp559[i0] = __tmp_in_tmp559;
}

auto tmp560 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp560 at (2223,1-2223,48) */
uint64_t __tmp_in_tmp560;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp560;
tmp560[i0][i1][i2][i3] = __tmp_in_tmp560;
}
}
}
}

auto tmp561 = make_vector<int64_t>( (int32_t)896);
/* Variable to read the clear value corresponding to the input variable tmp561 at (2225,1-2225,38) */
uint64_t __tmp_in_tmp561;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
cin >> __tmp_in_tmp561;
tmp561[i0] = __tmp_in_tmp561;
}

auto tmp562 = make_vector<int64_t>( (int32_t)896);
/* Variable to read the clear value corresponding to the input variable tmp562 at (2227,1-2227,38) */
uint64_t __tmp_in_tmp562;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
cin >> __tmp_in_tmp562;
tmp562[i0] = __tmp_in_tmp562;
}

auto tmp563 = make_vector<int64_t>( (int32_t)896);
/* Variable to read the clear value corresponding to the input variable tmp563 at (2229,1-2229,38) */
uint64_t __tmp_in_tmp563;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
cin >> __tmp_in_tmp563;
tmp563[i0] = __tmp_in_tmp563;
}

auto tmp564 = make_vector<int64_t>( (int32_t)896);
/* Variable to read the clear value corresponding to the input variable tmp564 at (2231,1-2231,38) */
uint64_t __tmp_in_tmp564;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
cin >> __tmp_in_tmp564;
tmp564[i0] = __tmp_in_tmp564;
}

auto tmp565 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)896,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp565 at (2233,1-2233,49) */
uint64_t __tmp_in_tmp565;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)896; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp565;
tmp565[i0][i1][i2][i3] = __tmp_in_tmp565;
}
}
}
}

auto tmp566 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp566 at (2235,1-2235,38) */
uint64_t __tmp_in_tmp566;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp566;
tmp566[i0] = __tmp_in_tmp566;
}

auto tmp567 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp567 at (2237,1-2237,38) */
uint64_t __tmp_in_tmp567;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp567;
tmp567[i0] = __tmp_in_tmp567;
}

auto tmp568 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp568 at (2239,1-2239,38) */
uint64_t __tmp_in_tmp568;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp568;
tmp568[i0] = __tmp_in_tmp568;
}

auto tmp569 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp569 at (2241,1-2241,38) */
uint64_t __tmp_in_tmp569;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp569;
tmp569[i0] = __tmp_in_tmp569;
}

auto tmp570 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp570 at (2243,1-2243,48) */
uint64_t __tmp_in_tmp570;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp570;
tmp570[i0][i1][i2][i3] = __tmp_in_tmp570;
}
}
}
}

auto tmp571 = make_vector<int64_t>( (int32_t)928);
/* Variable to read the clear value corresponding to the input variable tmp571 at (2245,1-2245,38) */
uint64_t __tmp_in_tmp571;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
cin >> __tmp_in_tmp571;
tmp571[i0] = __tmp_in_tmp571;
}

auto tmp572 = make_vector<int64_t>( (int32_t)928);
/* Variable to read the clear value corresponding to the input variable tmp572 at (2247,1-2247,38) */
uint64_t __tmp_in_tmp572;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
cin >> __tmp_in_tmp572;
tmp572[i0] = __tmp_in_tmp572;
}

auto tmp573 = make_vector<int64_t>( (int32_t)928);
/* Variable to read the clear value corresponding to the input variable tmp573 at (2249,1-2249,38) */
uint64_t __tmp_in_tmp573;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
cin >> __tmp_in_tmp573;
tmp573[i0] = __tmp_in_tmp573;
}

auto tmp574 = make_vector<int64_t>( (int32_t)928);
/* Variable to read the clear value corresponding to the input variable tmp574 at (2251,1-2251,38) */
uint64_t __tmp_in_tmp574;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
cin >> __tmp_in_tmp574;
tmp574[i0] = __tmp_in_tmp574;
}

auto tmp575 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)928,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp575 at (2253,1-2253,49) */
uint64_t __tmp_in_tmp575;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)928; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp575;
tmp575[i0][i1][i2][i3] = __tmp_in_tmp575;
}
}
}
}

auto tmp576 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp576 at (2255,1-2255,38) */
uint64_t __tmp_in_tmp576;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp576;
tmp576[i0] = __tmp_in_tmp576;
}

auto tmp577 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp577 at (2257,1-2257,38) */
uint64_t __tmp_in_tmp577;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp577;
tmp577[i0] = __tmp_in_tmp577;
}

auto tmp578 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp578 at (2259,1-2259,38) */
uint64_t __tmp_in_tmp578;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp578;
tmp578[i0] = __tmp_in_tmp578;
}

auto tmp579 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp579 at (2261,1-2261,38) */
uint64_t __tmp_in_tmp579;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp579;
tmp579[i0] = __tmp_in_tmp579;
}

auto tmp580 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp580 at (2263,1-2263,48) */
uint64_t __tmp_in_tmp580;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp580;
tmp580[i0][i1][i2][i3] = __tmp_in_tmp580;
}
}
}
}

auto tmp581 = make_vector<int64_t>( (int32_t)960);
/* Variable to read the clear value corresponding to the input variable tmp581 at (2265,1-2265,38) */
uint64_t __tmp_in_tmp581;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
cin >> __tmp_in_tmp581;
tmp581[i0] = __tmp_in_tmp581;
}

auto tmp582 = make_vector<int64_t>( (int32_t)960);
/* Variable to read the clear value corresponding to the input variable tmp582 at (2267,1-2267,38) */
uint64_t __tmp_in_tmp582;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
cin >> __tmp_in_tmp582;
tmp582[i0] = __tmp_in_tmp582;
}

auto tmp583 = make_vector<int64_t>( (int32_t)960);
/* Variable to read the clear value corresponding to the input variable tmp583 at (2269,1-2269,38) */
uint64_t __tmp_in_tmp583;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
cin >> __tmp_in_tmp583;
tmp583[i0] = __tmp_in_tmp583;
}

auto tmp584 = make_vector<int64_t>( (int32_t)960);
/* Variable to read the clear value corresponding to the input variable tmp584 at (2271,1-2271,38) */
uint64_t __tmp_in_tmp584;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
cin >> __tmp_in_tmp584;
tmp584[i0] = __tmp_in_tmp584;
}

auto tmp585 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)960,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp585 at (2273,1-2273,49) */
uint64_t __tmp_in_tmp585;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)960; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp585;
tmp585[i0][i1][i2][i3] = __tmp_in_tmp585;
}
}
}
}

auto tmp586 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp586 at (2275,1-2275,38) */
uint64_t __tmp_in_tmp586;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp586;
tmp586[i0] = __tmp_in_tmp586;
}

auto tmp587 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp587 at (2277,1-2277,38) */
uint64_t __tmp_in_tmp587;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp587;
tmp587[i0] = __tmp_in_tmp587;
}

auto tmp588 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp588 at (2279,1-2279,38) */
uint64_t __tmp_in_tmp588;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp588;
tmp588[i0] = __tmp_in_tmp588;
}

auto tmp589 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp589 at (2281,1-2281,38) */
uint64_t __tmp_in_tmp589;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp589;
tmp589[i0] = __tmp_in_tmp589;
}

auto tmp590 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp590 at (2283,1-2283,48) */
uint64_t __tmp_in_tmp590;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp590;
tmp590[i0][i1][i2][i3] = __tmp_in_tmp590;
}
}
}
}

auto tmp591 = make_vector<int64_t>( (int32_t)992);
/* Variable to read the clear value corresponding to the input variable tmp591 at (2285,1-2285,38) */
uint64_t __tmp_in_tmp591;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
cin >> __tmp_in_tmp591;
tmp591[i0] = __tmp_in_tmp591;
}

auto tmp592 = make_vector<int64_t>( (int32_t)992);
/* Variable to read the clear value corresponding to the input variable tmp592 at (2287,1-2287,38) */
uint64_t __tmp_in_tmp592;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
cin >> __tmp_in_tmp592;
tmp592[i0] = __tmp_in_tmp592;
}

auto tmp593 = make_vector<int64_t>( (int32_t)992);
/* Variable to read the clear value corresponding to the input variable tmp593 at (2289,1-2289,38) */
uint64_t __tmp_in_tmp593;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
cin >> __tmp_in_tmp593;
tmp593[i0] = __tmp_in_tmp593;
}

auto tmp594 = make_vector<int64_t>( (int32_t)992);
/* Variable to read the clear value corresponding to the input variable tmp594 at (2291,1-2291,38) */
uint64_t __tmp_in_tmp594;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
cin >> __tmp_in_tmp594;
tmp594[i0] = __tmp_in_tmp594;
}

auto tmp595 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)992,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp595 at (2293,1-2293,49) */
uint64_t __tmp_in_tmp595;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)992; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
cin >> __tmp_in_tmp595;
tmp595[i0][i1][i2][i3] = __tmp_in_tmp595;
}
}
}
}

auto tmp596 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp596 at (2295,1-2295,38) */
uint64_t __tmp_in_tmp596;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp596;
tmp596[i0] = __tmp_in_tmp596;
}

auto tmp597 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp597 at (2297,1-2297,38) */
uint64_t __tmp_in_tmp597;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp597;
tmp597[i0] = __tmp_in_tmp597;
}

auto tmp598 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp598 at (2299,1-2299,38) */
uint64_t __tmp_in_tmp598;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp598;
tmp598[i0] = __tmp_in_tmp598;
}

auto tmp599 = make_vector<int64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp599 at (2301,1-2301,38) */
uint64_t __tmp_in_tmp599;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
cin >> __tmp_in_tmp599;
tmp599[i0] = __tmp_in_tmp599;
}

auto tmp600 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32);
/* Variable to read the clear value corresponding to the input variable tmp600 at (2303,1-2303,48) */
uint64_t __tmp_in_tmp600;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
cin >> __tmp_in_tmp600;
tmp600[i0][i1][i2][i3] = __tmp_in_tmp600;
}
}
}
}

auto tmp601 = make_vector<int64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp601 at (2305,1-2305,39) */
uint64_t __tmp_in_tmp601;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp601;
tmp601[i0] = __tmp_in_tmp601;
}

auto tmp602 = make_vector<int64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp602 at (2307,1-2307,39) */
uint64_t __tmp_in_tmp602;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp602;
tmp602[i0] = __tmp_in_tmp602;
}

auto tmp603 = make_vector<int64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp603 at (2309,1-2309,39) */
uint64_t __tmp_in_tmp603;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp603;
tmp603[i0] = __tmp_in_tmp603;
}

auto tmp604 = make_vector<int64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp604 at (2311,1-2311,39) */
uint64_t __tmp_in_tmp604;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
cin >> __tmp_in_tmp604;
tmp604[i0] = __tmp_in_tmp604;
}

auto tmp605 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)1000);
/* Variable to read the clear value corresponding to the input variable tmp605 at (2313,1-2313,51) */
uint64_t __tmp_in_tmp605;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1000; i3++){
cin >> __tmp_in_tmp605;
tmp605[i0][i1][i2][i3] = __tmp_in_tmp605;
}
}
}
}

auto tmp606 = make_vector<int64_t>( (int32_t)1000);
/* Variable to read the clear value corresponding to the input variable tmp606 at (2315,1-2315,39) */
uint64_t __tmp_in_tmp606;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1000; i0++){
cin >> __tmp_in_tmp606;
tmp606[i0] = __tmp_in_tmp606;
}
Conv2DCSF( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3,  (int32_t)7,  (int32_t)7,  (int32_t)64,  (int32_t)2,  (int32_t)3,  (int32_t)2,  (int32_t)3,  (int32_t)2,  (int32_t)2, tmp0, tmp1, tmp607,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp607,  (int32_t)64, tmp2, tmp3, tmp608);
MaxPool( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)1,  (int32_t)0,  (int32_t)1,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp608, tmp609);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp609, tmp610);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp610,  (int32_t)64, tmp6, tmp7, tmp611);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp611, tmp612);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp612, tmp10, tmp613,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp613,  (int32_t)128, tmp11, tmp12, tmp614);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp614, tmp615);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp615, tmp15, tmp616,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp610,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp616,  (int32_t)3, tmp617);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96, tmp617,  (int32_t)96, tmp16, tmp17, tmp618);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96, tmp618, tmp619);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp619, tmp20, tmp620,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp620,  (int32_t)128, tmp21, tmp22, tmp621);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp621, tmp622);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp622, tmp25, tmp623,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96, tmp617,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp623,  (int32_t)3, tmp624);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp624,  (int32_t)128, tmp26, tmp27, tmp625);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp625, tmp626);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp626, tmp30, tmp627,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp627,  (int32_t)128, tmp31, tmp32, tmp628);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp628, tmp629);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp629, tmp35, tmp630,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp624,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp630,  (int32_t)3, tmp631);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160, tmp631,  (int32_t)160, tmp36, tmp37, tmp632);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160, tmp632, tmp633);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp633, tmp40, tmp634,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp634,  (int32_t)128, tmp41, tmp42, tmp635);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp635, tmp636);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp636, tmp45, tmp637,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160, tmp631,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp637,  (int32_t)3, tmp638);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192, tmp638,  (int32_t)192, tmp46, tmp47, tmp639);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192, tmp639, tmp640);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp640, tmp50, tmp641,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp641,  (int32_t)128, tmp51, tmp52, tmp642);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp642, tmp643);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp643, tmp55, tmp644,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192, tmp638,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp644,  (int32_t)3, tmp645);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224, tmp645,  (int32_t)224, tmp56, tmp57, tmp646);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224, tmp646, tmp647);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp647, tmp60, tmp648,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp648,  (int32_t)128, tmp61, tmp62, tmp649);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp649, tmp650);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp650, tmp65, tmp651,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224, tmp645,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp651,  (int32_t)3, tmp652);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp652,  (int32_t)256, tmp66, tmp67, tmp653);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp653, tmp654);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp654, tmp70, tmp655,  (int64_t)12);
AvgPool( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)2,  (int32_t)2,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp655, tmp656);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp656,  (int32_t)128, tmp71, tmp72, tmp657);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp657, tmp658);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp658, tmp75, tmp659,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp659,  (int32_t)128, tmp76, tmp77, tmp660);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp660, tmp661);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp661, tmp80, tmp662,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp656,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp662,  (int32_t)3, tmp663);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160, tmp663,  (int32_t)160, tmp81, tmp82, tmp664);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160, tmp664, tmp665);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp665, tmp85, tmp666,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp666,  (int32_t)128, tmp86, tmp87, tmp667);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp667, tmp668);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp668, tmp90, tmp669,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160, tmp663,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp669,  (int32_t)3, tmp670);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192, tmp670,  (int32_t)192, tmp91, tmp92, tmp671);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192, tmp671, tmp672);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp672, tmp95, tmp673,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp673,  (int32_t)128, tmp96, tmp97, tmp674);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp674, tmp675);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp675, tmp100, tmp676,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192, tmp670,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp676,  (int32_t)3, tmp677);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224, tmp677,  (int32_t)224, tmp101, tmp102, tmp678);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224, tmp678, tmp679);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp679, tmp105, tmp680,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp680,  (int32_t)128, tmp106, tmp107, tmp681);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp681, tmp682);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp682, tmp110, tmp683,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224, tmp677,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp683,  (int32_t)3, tmp684);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp684,  (int32_t)256, tmp111, tmp112, tmp685);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp685, tmp686);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp686, tmp115, tmp687,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp687,  (int32_t)128, tmp116, tmp117, tmp688);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp688, tmp689);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp689, tmp120, tmp690,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp684,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp690,  (int32_t)3, tmp691);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288, tmp691,  (int32_t)288, tmp121, tmp122, tmp692);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288, tmp692, tmp693);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp693, tmp125, tmp694,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp694,  (int32_t)128, tmp126, tmp127, tmp695);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp695, tmp696);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp696, tmp130, tmp697,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288, tmp691,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp697,  (int32_t)3, tmp698);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320, tmp698,  (int32_t)320, tmp131, tmp132, tmp699);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320, tmp699, tmp700);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp700, tmp135, tmp701,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp701,  (int32_t)128, tmp136, tmp137, tmp702);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp702, tmp703);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp703, tmp140, tmp704,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320, tmp698,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp704,  (int32_t)3, tmp705);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352, tmp705,  (int32_t)352, tmp141, tmp142, tmp706);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352, tmp706, tmp707);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp707, tmp145, tmp708,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp708,  (int32_t)128, tmp146, tmp147, tmp709);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp709, tmp710);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp710, tmp150, tmp711,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352, tmp705,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp711,  (int32_t)3, tmp712);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384, tmp712,  (int32_t)384, tmp151, tmp152, tmp713);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384, tmp713, tmp714);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp714, tmp155, tmp715,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp715,  (int32_t)128, tmp156, tmp157, tmp716);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp716, tmp717);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp717, tmp160, tmp718,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384, tmp712,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp718,  (int32_t)3, tmp719);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416, tmp719,  (int32_t)416, tmp161, tmp162, tmp720);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416, tmp720, tmp721);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp721, tmp165, tmp722,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp722,  (int32_t)128, tmp166, tmp167, tmp723);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp723, tmp724);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp724, tmp170, tmp725,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416, tmp719,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp725,  (int32_t)3, tmp726);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448, tmp726,  (int32_t)448, tmp171, tmp172, tmp727);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448, tmp727, tmp728);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp728, tmp175, tmp729,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp729,  (int32_t)128, tmp176, tmp177, tmp730);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp730, tmp731);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp731, tmp180, tmp732,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448, tmp726,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp732,  (int32_t)3, tmp733);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480, tmp733,  (int32_t)480, tmp181, tmp182, tmp734);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480, tmp734, tmp735);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp735, tmp185, tmp736,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp736,  (int32_t)128, tmp186, tmp187, tmp737);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp737, tmp738);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp738, tmp190, tmp739,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480, tmp733,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp739,  (int32_t)3, tmp740);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp740,  (int32_t)512, tmp191, tmp192, tmp741);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp741, tmp742);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp742, tmp195, tmp743,  (int64_t)12);
AvgPool( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)2,  (int32_t)2,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp743, tmp744);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp744,  (int32_t)256, tmp196, tmp197, tmp745);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp745, tmp746);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp746, tmp200, tmp747,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp747,  (int32_t)128, tmp201, tmp202, tmp748);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp748, tmp749);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp749, tmp205, tmp750,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp744,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp750,  (int32_t)3, tmp751);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288, tmp751,  (int32_t)288, tmp206, tmp207, tmp752);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288, tmp752, tmp753);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp753, tmp210, tmp754,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp754,  (int32_t)128, tmp211, tmp212, tmp755);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp755, tmp756);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp756, tmp215, tmp757,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288, tmp751,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp757,  (int32_t)3, tmp758);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320, tmp758,  (int32_t)320, tmp216, tmp217, tmp759);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320, tmp759, tmp760);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp760, tmp220, tmp761,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp761,  (int32_t)128, tmp221, tmp222, tmp762);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp762, tmp763);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp763, tmp225, tmp764,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320, tmp758,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp764,  (int32_t)3, tmp765);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352, tmp765,  (int32_t)352, tmp226, tmp227, tmp766);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352, tmp766, tmp767);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp767, tmp230, tmp768,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp768,  (int32_t)128, tmp231, tmp232, tmp769);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp769, tmp770);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp770, tmp235, tmp771,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352, tmp765,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp771,  (int32_t)3, tmp772);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384, tmp772,  (int32_t)384, tmp236, tmp237, tmp773);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384, tmp773, tmp774);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp774, tmp240, tmp775,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp775,  (int32_t)128, tmp241, tmp242, tmp776);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp776, tmp777);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp777, tmp245, tmp778,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384, tmp772,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp778,  (int32_t)3, tmp779);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416, tmp779,  (int32_t)416, tmp246, tmp247, tmp780);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416, tmp780, tmp781);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp781, tmp250, tmp782,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp782,  (int32_t)128, tmp251, tmp252, tmp783);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp783, tmp784);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp784, tmp255, tmp785,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416, tmp779,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp785,  (int32_t)3, tmp786);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448, tmp786,  (int32_t)448, tmp256, tmp257, tmp787);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448, tmp787, tmp788);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp788, tmp260, tmp789,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp789,  (int32_t)128, tmp261, tmp262, tmp790);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp790, tmp791);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp791, tmp265, tmp792,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448, tmp786,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp792,  (int32_t)3, tmp793);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480, tmp793,  (int32_t)480, tmp266, tmp267, tmp794);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480, tmp794, tmp795);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp795, tmp270, tmp796,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp796,  (int32_t)128, tmp271, tmp272, tmp797);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp797, tmp798);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp798, tmp275, tmp799,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480, tmp793,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp799,  (int32_t)3, tmp800);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp800,  (int32_t)512, tmp276, tmp277, tmp801);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp801, tmp802);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp802, tmp280, tmp803,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp803,  (int32_t)128, tmp281, tmp282, tmp804);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp804, tmp805);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp805, tmp285, tmp806,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp800,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp806,  (int32_t)3, tmp807);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544, tmp807,  (int32_t)544, tmp286, tmp287, tmp808);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544, tmp808, tmp809);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp809, tmp290, tmp810,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp810,  (int32_t)128, tmp291, tmp292, tmp811);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp811, tmp812);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp812, tmp295, tmp813,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544, tmp807,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp813,  (int32_t)3, tmp814);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576, tmp814,  (int32_t)576, tmp296, tmp297, tmp815);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576, tmp815, tmp816);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp816, tmp300, tmp817,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp817,  (int32_t)128, tmp301, tmp302, tmp818);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp818, tmp819);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp819, tmp305, tmp820,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576, tmp814,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp820,  (int32_t)3, tmp821);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608, tmp821,  (int32_t)608, tmp306, tmp307, tmp822);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608, tmp822, tmp823);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp823, tmp310, tmp824,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp824,  (int32_t)128, tmp311, tmp312, tmp825);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp825, tmp826);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp826, tmp315, tmp827,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608, tmp821,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp827,  (int32_t)3, tmp828);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640, tmp828,  (int32_t)640, tmp316, tmp317, tmp829);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640, tmp829, tmp830);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp830, tmp320, tmp831,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp831,  (int32_t)128, tmp321, tmp322, tmp832);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp832, tmp833);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp833, tmp325, tmp834,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640, tmp828,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp834,  (int32_t)3, tmp835);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672, tmp835,  (int32_t)672, tmp326, tmp327, tmp836);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672, tmp836, tmp837);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp837, tmp330, tmp838,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp838,  (int32_t)128, tmp331, tmp332, tmp839);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp839, tmp840);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp840, tmp335, tmp841,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672, tmp835,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp841,  (int32_t)3, tmp842);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704, tmp842,  (int32_t)704, tmp336, tmp337, tmp843);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704, tmp843, tmp844);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp844, tmp340, tmp845,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp845,  (int32_t)128, tmp341, tmp342, tmp846);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp846, tmp847);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp847, tmp345, tmp848,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704, tmp842,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp848,  (int32_t)3, tmp849);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736, tmp849,  (int32_t)736, tmp346, tmp347, tmp850);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736, tmp850, tmp851);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp851, tmp350, tmp852,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp852,  (int32_t)128, tmp351, tmp352, tmp853);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp853, tmp854);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp854, tmp355, tmp855,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736, tmp849,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp855,  (int32_t)3, tmp856);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768, tmp856,  (int32_t)768, tmp356, tmp357, tmp857);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768, tmp857, tmp858);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp858, tmp360, tmp859,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp859,  (int32_t)128, tmp361, tmp362, tmp860);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp860, tmp861);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp861, tmp365, tmp862,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768, tmp856,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp862,  (int32_t)3, tmp863);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800, tmp863,  (int32_t)800, tmp366, tmp367, tmp864);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800, tmp864, tmp865);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp865, tmp370, tmp866,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp866,  (int32_t)128, tmp371, tmp372, tmp867);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp867, tmp868);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp868, tmp375, tmp869,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800, tmp863,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp869,  (int32_t)3, tmp870);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832, tmp870,  (int32_t)832, tmp376, tmp377, tmp871);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832, tmp871, tmp872);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp872, tmp380, tmp873,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp873,  (int32_t)128, tmp381, tmp382, tmp874);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp874, tmp875);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp875, tmp385, tmp876,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832, tmp870,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp876,  (int32_t)3, tmp877);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864, tmp877,  (int32_t)864, tmp386, tmp387, tmp878);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864, tmp878, tmp879);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp879, tmp390, tmp880,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp880,  (int32_t)128, tmp391, tmp392, tmp881);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp881, tmp882);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp882, tmp395, tmp883,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864, tmp877,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp883,  (int32_t)3, tmp884);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896, tmp884,  (int32_t)896, tmp396, tmp397, tmp885);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896, tmp885, tmp886);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp886, tmp400, tmp887,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp887,  (int32_t)128, tmp401, tmp402, tmp888);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp888, tmp889);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp889, tmp405, tmp890,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896, tmp884,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp890,  (int32_t)3, tmp891);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928, tmp891,  (int32_t)928, tmp406, tmp407, tmp892);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928, tmp892, tmp893);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp893, tmp410, tmp894,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp894,  (int32_t)128, tmp411, tmp412, tmp895);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp895, tmp896);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp896, tmp415, tmp897,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928, tmp891,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp897,  (int32_t)3, tmp898);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960, tmp898,  (int32_t)960, tmp416, tmp417, tmp899);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960, tmp899, tmp900);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp900, tmp420, tmp901,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp901,  (int32_t)128, tmp421, tmp422, tmp902);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp902, tmp903);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp903, tmp425, tmp904,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960, tmp898,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp904,  (int32_t)3, tmp905);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992, tmp905,  (int32_t)992, tmp426, tmp427, tmp906);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992, tmp906, tmp907);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp907, tmp430, tmp908,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp908,  (int32_t)128, tmp431, tmp432, tmp909);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp909, tmp910);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp910, tmp435, tmp911,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992, tmp905,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp911,  (int32_t)3, tmp912);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp912,  (int32_t)1024, tmp436, tmp437, tmp913);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp913, tmp914);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp914, tmp440, tmp915,  (int64_t)12);
AvgPool( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)2,  (int32_t)2,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp915, tmp916);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp916,  (int32_t)512, tmp441, tmp442, tmp917);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp917, tmp918);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp918, tmp445, tmp919,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp919,  (int32_t)128, tmp446, tmp447, tmp920);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp920, tmp921);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp921, tmp450, tmp922,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp916,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp922,  (int32_t)3, tmp923);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544, tmp923,  (int32_t)544, tmp451, tmp452, tmp924);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544, tmp924, tmp925);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp925, tmp455, tmp926,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp926,  (int32_t)128, tmp456, tmp457, tmp927);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp927, tmp928);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp928, tmp460, tmp929,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544, tmp923,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp929,  (int32_t)3, tmp930);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576, tmp930,  (int32_t)576, tmp461, tmp462, tmp931);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576, tmp931, tmp932);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp932, tmp465, tmp933,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp933,  (int32_t)128, tmp466, tmp467, tmp934);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp934, tmp935);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp935, tmp470, tmp936,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576, tmp930,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp936,  (int32_t)3, tmp937);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608, tmp937,  (int32_t)608, tmp471, tmp472, tmp938);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608, tmp938, tmp939);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp939, tmp475, tmp940,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp940,  (int32_t)128, tmp476, tmp477, tmp941);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp941, tmp942);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp942, tmp480, tmp943,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608, tmp937,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp943,  (int32_t)3, tmp944);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640, tmp944,  (int32_t)640, tmp481, tmp482, tmp945);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640, tmp945, tmp946);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp946, tmp485, tmp947,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp947,  (int32_t)128, tmp486, tmp487, tmp948);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp948, tmp949);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp949, tmp490, tmp950,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640, tmp944,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp950,  (int32_t)3, tmp951);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672, tmp951,  (int32_t)672, tmp491, tmp492, tmp952);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672, tmp952, tmp953);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp953, tmp495, tmp954,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp954,  (int32_t)128, tmp496, tmp497, tmp955);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp955, tmp956);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp956, tmp500, tmp957,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672, tmp951,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp957,  (int32_t)3, tmp958);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704, tmp958,  (int32_t)704, tmp501, tmp502, tmp959);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704, tmp959, tmp960);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp960, tmp505, tmp961,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp961,  (int32_t)128, tmp506, tmp507, tmp962);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp962, tmp963);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp963, tmp510, tmp964,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704, tmp958,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp964,  (int32_t)3, tmp965);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736, tmp965,  (int32_t)736, tmp511, tmp512, tmp966);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736, tmp966, tmp967);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp967, tmp515, tmp968,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp968,  (int32_t)128, tmp516, tmp517, tmp969);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp969, tmp970);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp970, tmp520, tmp971,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736, tmp965,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp971,  (int32_t)3, tmp972);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768, tmp972,  (int32_t)768, tmp521, tmp522, tmp973);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768, tmp973, tmp974);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp974, tmp525, tmp975,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp975,  (int32_t)128, tmp526, tmp527, tmp976);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp976, tmp977);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp977, tmp530, tmp978,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768, tmp972,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp978,  (int32_t)3, tmp979);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800, tmp979,  (int32_t)800, tmp531, tmp532, tmp980);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800, tmp980, tmp981);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp981, tmp535, tmp982,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp982,  (int32_t)128, tmp536, tmp537, tmp983);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp983, tmp984);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp984, tmp540, tmp985,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800, tmp979,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp985,  (int32_t)3, tmp986);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832, tmp986,  (int32_t)832, tmp541, tmp542, tmp987);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832, tmp987, tmp988);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp988, tmp545, tmp989,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp989,  (int32_t)128, tmp546, tmp547, tmp990);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp990, tmp991);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp991, tmp550, tmp992,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832, tmp986,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp992,  (int32_t)3, tmp993);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864, tmp993,  (int32_t)864, tmp551, tmp552, tmp994);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864, tmp994, tmp995);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp995, tmp555, tmp996,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp996,  (int32_t)128, tmp556, tmp557, tmp997);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp997, tmp998);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp998, tmp560, tmp999,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864, tmp993,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp999,  (int32_t)3, tmp1000);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896, tmp1000,  (int32_t)896, tmp561, tmp562, tmp1001);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896, tmp1001, tmp1002);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1002, tmp565, tmp1003,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1003,  (int32_t)128, tmp566, tmp567, tmp1004);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1004, tmp1005);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1005, tmp570, tmp1006,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896, tmp1000,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1006,  (int32_t)3, tmp1007);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928, tmp1007,  (int32_t)928, tmp571, tmp572, tmp1008);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928, tmp1008, tmp1009);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1009, tmp575, tmp1010,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1010,  (int32_t)128, tmp576, tmp577, tmp1011);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1011, tmp1012);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1012, tmp580, tmp1013,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928, tmp1007,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1013,  (int32_t)3, tmp1014);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960, tmp1014,  (int32_t)960, tmp581, tmp582, tmp1015);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960, tmp1015, tmp1016);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1016, tmp585, tmp1017,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1017,  (int32_t)128, tmp586, tmp587, tmp1018);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1018, tmp1019);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1019, tmp590, tmp1020,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960, tmp1014,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1020,  (int32_t)3, tmp1021);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992, tmp1021,  (int32_t)992, tmp591, tmp592, tmp1022);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992, tmp1022, tmp1023);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1023, tmp595, tmp1024,  (int64_t)12);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1024,  (int32_t)128, tmp596, tmp597, tmp1025);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1025, tmp1026);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1026, tmp600, tmp1027,  (int64_t)12);
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992, tmp1021,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1027,  (int32_t)3, tmp1028);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024, tmp1028,  (int32_t)1024, tmp601, tmp602, tmp1029);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024, tmp1029, tmp1030);
AvgPool( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)7,  (int32_t)7,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024, tmp1030, tmp1031);
Conv2DCSF( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)1000,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1031, tmp605, tmp1032,  (int64_t)12);
MatAddBroadCast4( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000, tmp1032, tmp606, tmp1033);
// cout<<(tmp1033) << endl;
return 0;
}

