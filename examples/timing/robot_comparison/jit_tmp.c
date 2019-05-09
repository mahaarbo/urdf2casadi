/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CODEGEN_PREFIX
  #define NAMESPACE_CONCAT(NS, ID) _NAMESPACE_CONCAT(NS, ID)
  #define _NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) jit_tmp_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};

casadi_real casadi_sq(casadi_real x) { return x*x;}

/* q_ddot:(i0[6],i1[6],i2[6])->(o0[6]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=1.0267495893000000e-02;
  a1=2.2689067591000001e-01;
  a2=arg[0] ? arg[0][2] : 0;
  a3=cos(a2);
  a4=4.9443313555999999e-02;
  a5=1.1117275553100001e-01;
  a6=3.;
  a7=arg[0] ? arg[0][5] : 0;
  a8=sin(a7);
  a9=(a6-a8);
  a7=cos(a7);
  a9=(a9+a7);
  a10=-1.;
  a11=(a10+a8);
  a12=(a8*a11);
  a9=(a9+a12);
  a12=1.;
  a13=(a12+a7);
  a14=(a7*a13);
  a9=(a9+a14);
  a9=(a5+a9);
  a14=(a8*a7);
  a15=(a7+a8);
  a14=(a14-a15);
  a15=(a7*a8);
  a14=(a14-a15);
  a15=arg[0] ? arg[0][4] : 0;
  a16=sin(a15);
  a17=(a14*a16);
  a17=(a9+a17);
  a18=(a7-a8);
  a19=casadi_sq(a8);
  a18=(a18+a19);
  a19=casadi_sq(a7);
  a18=(a18+a19);
  a15=cos(a15);
  a19=(a18*a15);
  a17=(a17+a19);
  a19=(a7*a11);
  a20=(a8*a13);
  a19=(a19-a20);
  a20=2.2190000000000003e+00;
  a21=(a20*a16);
  a21=(a19+a21);
  a22=(a16*a21);
  a17=(a17+a22);
  a11=(a8*a11);
  a13=(a7*a13);
  a11=(a11+a13);
  a13=(a20*a15);
  a13=(a11+a13);
  a22=(a15*a13);
  a17=(a17+a22);
  a17=(a5+a17);
  a22=(a14*a15);
  a23=(a18*a16);
  a22=(a22-a23);
  a23=(a20*a15);
  a24=(a16*a23);
  a22=(a22+a24);
  a24=(a20*a16);
  a25=(a15*a24);
  a22=(a22-a25);
  a25=-5.0000000000000000e-01;
  a26=arg[0] ? arg[0][3] : 0;
  a27=sin(a26);
  a28=(a25*a27);
  a22=(a22*a28);
  a22=(a17-a22);
  a14=(a14*a16);
  a18=(a18*a15);
  a14=(a14+a18);
  a18=(a20*a16);
  a29=(a16*a18);
  a14=(a14+a29);
  a29=(a20*a15);
  a30=(a15*a29);
  a14=(a14+a30);
  a26=cos(a26);
  a30=(a25*a26);
  a14=(a14*a30);
  a22=(a22-a14);
  a14=(a15*a21);
  a31=(a16*a13);
  a14=(a14-a31);
  a31=1.2190000000000001e+00;
  a32=(a15*a23);
  a33=(a16*a24);
  a32=(a32+a33);
  a32=(a31+a32);
  a32=(a32*a28);
  a32=(a14-a32);
  a33=(a15*a18);
  a34=(a16*a29);
  a33=(a33-a34);
  a33=(a33*a30);
  a32=(a32-a33);
  a32=(a28*a32);
  a22=(a22-a32);
  a21=(a16*a21);
  a13=(a15*a13);
  a21=(a21+a13);
  a23=(a16*a23);
  a24=(a15*a24);
  a23=(a23-a24);
  a23=(a23*a28);
  a23=(a21-a23);
  a18=(a16*a18);
  a29=(a15*a29);
  a18=(a18+a29);
  a18=(a31+a18);
  a18=(a18*a30);
  a23=(a23-a18);
  a23=(a30*a23);
  a22=(a22-a23);
  a22=(a4+a22);
  a22=(a22*a3);
  a23=(a8-a7);
  a18=casadi_sq(a8);
  a23=(a23-a18);
  a18=casadi_sq(a7);
  a23=(a23-a18);
  a18=(a23*a15);
  a29=(a8*a7);
  a24=(a8+a7);
  a29=(a29-a24);
  a24=(a7*a8);
  a29=(a29-a24);
  a24=(a29*a16);
  a18=(a18-a24);
  a24=(a20*a16);
  a13=(a16*a24);
  a18=(a18-a13);
  a13=-2.2190000000000003e+00;
  a32=(a13*a15);
  a33=(a15*a32);
  a18=(a18+a33);
  a33=(a18*a27);
  a23=(a23*a16);
  a29=(a29*a15);
  a23=(a23+a29);
  a29=(a20*a15);
  a34=(a16*a29);
  a23=(a23+a34);
  a34=(a13*a16);
  a35=(a15*a34);
  a23=(a23+a35);
  a35=(a23*a26);
  a33=(a33+a35);
  a35=(a15*a29);
  a36=(a16*a34);
  a35=(a35-a36);
  a36=(a35*a26);
  a37=(a15*a24);
  a38=(a16*a32);
  a37=(a37+a38);
  a38=(a37*a27);
  a36=(a36-a38);
  a36=(a28*a36);
  a33=(a33-a36);
  a32=(a15*a32);
  a24=(a16*a24);
  a32=(a32-a24);
  a24=(a32*a27);
  a29=(a16*a29);
  a34=(a15*a34);
  a29=(a29+a34);
  a34=(a29*a26);
  a24=(a24+a34);
  a24=(a30*a24);
  a33=(a33-a24);
  a2=sin(a2);
  a33=(a33*a2);
  a22=(a22+a33);
  a22=(a3*a22);
  a33=(a10-a7);
  a24=(a7*a33);
  a34=(a10+a8);
  a36=(a8*a34);
  a24=(a24-a36);
  a36=(a13*a15);
  a36=(a24+a36);
  a38=(a15*a36);
  a33=(a8*a33);
  a34=(a7*a34);
  a33=(a33+a34);
  a34=(a20*a16);
  a34=(a33+a34);
  a39=(a16*a34);
  a38=(a38-a39);
  a39=(a13*a16);
  a40=(a15*a39);
  a41=(a20*a15);
  a42=(a16*a41);
  a40=(a40+a42);
  a40=(a40*a28);
  a40=(a38+a40);
  a42=(a13*a15);
  a43=(a15*a42);
  a20=(a20*a16);
  a44=(a16*a20);
  a43=(a43-a44);
  a43=(a43*a30);
  a40=(a40-a43);
  a40=(a27*a40);
  a36=(a16*a36);
  a34=(a15*a34);
  a36=(a36+a34);
  a41=(a15*a41);
  a39=(a16*a39);
  a41=(a41-a39);
  a41=(a41*a28);
  a41=(a36-a41);
  a42=(a16*a42);
  a20=(a15*a20);
  a42=(a42+a20);
  a42=(a42*a30);
  a41=(a41-a42);
  a41=(a26*a41);
  a40=(a40+a41);
  a40=(a40*a3);
  a41=2.6845000000000001e-02;
  a42=1.3301727555310001e+00;
  a20=(a6*a7);
  a20=(a20+a8);
  a39=(a7*a20);
  a34=(a6*a8);
  a34=(a7+a34);
  a43=(a8*a34);
  a39=(a39+a43);
  a39=(a42+a39);
  a43=(a39*a15);
  a44=(a6*a8);
  a44=(a44-a7);
  a44=(a44+a10);
  a45=(a7*a44);
  a46=(a6*a7);
  a46=(a46-a8);
  a46=(a46+a12);
  a12=(a8*a46);
  a45=(a45-a12);
  a12=(a45*a16);
  a43=(a43-a12);
  a12=(a15*a43);
  a20=(a8*a20);
  a34=(a7*a34);
  a20=(a20-a34);
  a34=(a7+a8);
  a20=(a20-a34);
  a47=(a20*a15);
  a48=1.4384200000000000e+00;
  a44=(a8*a44);
  a46=(a7*a46);
  a44=(a44+a46);
  a46=(a8-a7);
  a46=(a46+a10);
  a44=(a44-a46);
  a44=(a48+a44);
  a49=(a44*a16);
  a47=(a47-a49);
  a49=(a16*a47);
  a12=(a12-a49);
  a12=(a5+a12);
  a49=(a12*a27);
  a39=(a39*a16);
  a45=(a45*a15);
  a39=(a39+a45);
  a45=(a7+a8);
  a39=(a39-a45);
  a50=(a15*a39);
  a20=(a20*a16);
  a44=(a44*a15);
  a20=(a20+a44);
  a44=(a8-a7);
  a44=(a44+a10);
  a20=(a20-a44);
  a10=(a16*a20);
  a50=(a50-a10);
  a10=(a50*a26);
  a49=(a49+a10);
  a10=(a15*a45);
  a51=(a16*a44);
  a10=(a10-a51);
  a10=(a25*a10);
  a49=(a49+a10);
  a49=(a27*a49);
  a43=(a16*a43);
  a47=(a15*a47);
  a43=(a43+a47);
  a47=(a34*a15);
  a10=(a46*a16);
  a47=(a47-a10);
  a43=(a43-a47);
  a10=(a43*a27);
  a51=2.1942000000000000e-01;
  a39=(a16*a39);
  a20=(a15*a20);
  a39=(a39+a20);
  a34=(a34*a16);
  a46=(a46*a15);
  a34=(a34+a46);
  a34=(a34+a13);
  a39=(a39-a34);
  a39=(a51+a39);
  a46=(a39*a26);
  a10=(a10+a46);
  a45=(a16*a45);
  a44=(a15*a44);
  a45=(a45+a44);
  a45=(a45+a13);
  a45=(a25*a45);
  a10=(a10+a45);
  a10=(a26*a10);
  a49=(a49+a10);
  a10=(a47*a27);
  a45=(a34*a26);
  a10=(a10+a45);
  a45=-1.7190000000000003e+00;
  a10=(a10+a45);
  a10=(a25*a10);
  a49=(a49+a10);
  a49=(a41+a49);
  a49=(a49*a2);
  a40=(a40+a49);
  a40=(a2*a40);
  a22=(a22+a40);
  a1=(a1+a22);
  a0=(a0+a1);
  a22=casadi_sq(a0);
  a40=casadi_sq(a1);
  a22=(a22+a40);
  a18=(a18*a26);
  a23=(a23*a27);
  a18=(a18-a23);
  a37=(a37*a26);
  a35=(a35*a27);
  a37=(a37+a35);
  a37=(a28*a37);
  a18=(a18+a37);
  a32=(a32*a26);
  a29=(a29*a27);
  a32=(a32-a29);
  a32=(a30*a32);
  a18=(a18-a32);
  a18=(a3*a18);
  a12=(a12*a26);
  a50=(a50*a27);
  a12=(a12-a50);
  a50=(a27*a12);
  a43=(a43*a26);
  a39=(a39*a27);
  a43=(a43-a39);
  a39=(a26*a43);
  a50=(a50+a39);
  a47=(a47*a26);
  a34=(a34*a27);
  a47=(a47-a34);
  a47=(a25*a47);
  a50=(a50+a47);
  a50=(a2*a50);
  a18=(a18+a50);
  a50=casadi_sq(a18);
  a22=(a22+a50);
  a14=(a28*a14);
  a14=(a17-a14);
  a21=(a30*a21);
  a14=(a14-a21);
  a14=(a3*a14);
  a21=(a27*a38);
  a50=(a26*a36);
  a21=(a21+a50);
  a21=(a2*a21);
  a14=(a14+a21);
  a21=casadi_sq(a14);
  a22=(a22+a21);
  a21=(a16*a19);
  a21=(a9+a21);
  a50=(a15*a11);
  a21=(a21+a50);
  a50=(a15*a19);
  a47=(a16*a11);
  a50=(a50-a47);
  a50=(a28*a50);
  a50=(a21-a50);
  a19=(a16*a19);
  a11=(a15*a11);
  a19=(a19+a11);
  a19=(a30*a19);
  a50=(a50-a19);
  a50=(a3*a50);
  a19=(a15*a24);
  a11=(a16*a33);
  a19=(a19-a11);
  a11=(a27*a19);
  a24=(a16*a24);
  a33=(a15*a33);
  a24=(a24+a33);
  a33=(a26*a24);
  a11=(a11+a33);
  a11=(a2*a11);
  a50=(a50+a11);
  a11=casadi_sq(a50);
  a22=(a22+a11);
  a11=(a6-a8);
  a11=(a11+a7);
  a33=(a7+a8);
  a47=(a16*a33);
  a47=(a11-a47);
  a34=(a7-a8);
  a39=(a15*a34);
  a47=(a47+a39);
  a39=(a15*a33);
  a32=(a16*a34);
  a39=(a39+a32);
  a39=(a28*a39);
  a39=(a47+a39);
  a34=(a15*a34);
  a33=(a16*a33);
  a34=(a34-a33);
  a34=(a30*a34);
  a39=(a39-a34);
  a39=(a3*a39);
  a34=(a8-a7);
  a33=(a15*a34);
  a32=(a8+a7);
  a29=(a16*a32);
  a33=(a33+a29);
  a29=(a27*a33);
  a34=(a16*a34);
  a32=(a15*a32);
  a34=(a34-a32);
  a32=(a26*a34);
  a29=(a29+a32);
  a29=(a2*a29);
  a39=(a39+a29);
  a29=casadi_sq(a39);
  a22=(a22+a29);
  a22=sqrt(a22);
  a0=(a0/a22);
  a29=(a39*a0);
  a32=(a1/a22);
  a37=(a39*a32);
  a29=(a29+a37);
  a33=(a26*a33);
  a34=(a27*a34);
  a33=(a33-a34);
  a34=(a18/a22);
  a37=(a33*a34);
  a29=(a29+a37);
  a37=(a14/a22);
  a35=(a47*a37);
  a29=(a29+a35);
  a35=(a50/a22);
  a23=(a11*a35);
  a29=(a29+a23);
  a23=(a39/a22);
  a40=(a6*a23);
  a29=(a29+a40);
  a40=(a29*a0);
  a40=(a39-a40);
  a49=(a1*a0);
  a10=(a1*a32);
  a49=(a49+a10);
  a10=(a18*a34);
  a49=(a49+a10);
  a10=(a14*a37);
  a49=(a49+a10);
  a10=(a50*a35);
  a49=(a49+a10);
  a10=(a39*a23);
  a49=(a49+a10);
  a10=(a49*a0);
  a10=(a1-a10);
  a45=casadi_sq(a10);
  a13=(a49*a32);
  a1=(a1-a13);
  a13=casadi_sq(a1);
  a45=(a45+a13);
  a13=(a49*a34);
  a13=(a18-a13);
  a44=casadi_sq(a13);
  a45=(a45+a44);
  a44=(a49*a37);
  a44=(a14-a44);
  a46=casadi_sq(a44);
  a45=(a45+a46);
  a46=(a49*a35);
  a46=(a50-a46);
  a20=casadi_sq(a46);
  a45=(a45+a20);
  a20=(a49*a23);
  a20=(a39-a20);
  a52=casadi_sq(a20);
  a45=(a45+a52);
  a45=sqrt(a45);
  a10=(a10/a45);
  a52=(a40*a10);
  a53=(a29*a32);
  a39=(a39-a53);
  a1=(a1/a45);
  a53=(a39*a1);
  a52=(a52+a53);
  a53=(a29*a34);
  a53=(a33-a53);
  a13=(a13/a45);
  a54=(a53*a13);
  a52=(a52+a54);
  a54=(a29*a37);
  a54=(a47-a54);
  a44=(a44/a45);
  a55=(a54*a44);
  a52=(a52+a55);
  a55=(a29*a35);
  a55=(a11-a55);
  a46=(a46/a45);
  a56=(a55*a46);
  a52=(a52+a56);
  a56=(a29*a23);
  a56=(a6-a56);
  a20=(a20/a45);
  a57=(a56*a20);
  a52=(a52+a57);
  a57=(a52*a10);
  a40=(a40-a57);
  a57=(a18*a0);
  a58=(a18*a32);
  a57=(a57+a58);
  a58=7.2193313556000005e-02;
  a12=(a26*a12);
  a43=(a27*a43);
  a12=(a12-a43);
  a12=(a58+a12);
  a43=(a12*a34);
  a57=(a57+a43);
  a38=(a26*a38);
  a36=(a27*a36);
  a38=(a38-a36);
  a36=(a38*a37);
  a57=(a57+a36);
  a19=(a26*a19);
  a24=(a27*a24);
  a19=(a19-a24);
  a24=(a19*a35);
  a57=(a57+a24);
  a24=(a33*a23);
  a57=(a57+a24);
  a24=(a57*a0);
  a24=(a18-a24);
  a36=(a24*a10);
  a43=(a57*a32);
  a18=(a18-a43);
  a43=(a18*a1);
  a36=(a36+a43);
  a43=(a57*a34);
  a12=(a12-a43);
  a43=(a12*a13);
  a36=(a36+a43);
  a43=(a57*a37);
  a43=(a38-a43);
  a59=(a43*a44);
  a36=(a36+a59);
  a59=(a57*a35);
  a59=(a19-a59);
  a60=(a59*a46);
  a36=(a36+a60);
  a60=(a57*a23);
  a33=(a33-a60);
  a60=(a33*a20);
  a36=(a36+a60);
  a60=(a36*a10);
  a24=(a24-a60);
  a60=casadi_sq(a24);
  a61=(a36*a1);
  a18=(a18-a61);
  a61=casadi_sq(a18);
  a60=(a60+a61);
  a61=(a36*a13);
  a12=(a12-a61);
  a61=casadi_sq(a12);
  a60=(a60+a61);
  a61=(a36*a44);
  a43=(a43-a61);
  a61=casadi_sq(a43);
  a60=(a60+a61);
  a61=(a36*a46);
  a59=(a59-a61);
  a61=casadi_sq(a59);
  a60=(a60+a61);
  a61=(a36*a20);
  a33=(a33-a61);
  a61=casadi_sq(a33);
  a60=(a60+a61);
  a60=sqrt(a60);
  a24=(a24/a60);
  a61=(a40*a24);
  a62=(a52*a1);
  a39=(a39-a62);
  a18=(a18/a60);
  a62=(a39*a18);
  a61=(a61+a62);
  a62=(a52*a13);
  a53=(a53-a62);
  a12=(a12/a60);
  a62=(a53*a12);
  a61=(a61+a62);
  a62=(a52*a44);
  a54=(a54-a62);
  a43=(a43/a60);
  a62=(a54*a43);
  a61=(a61+a62);
  a62=(a52*a46);
  a55=(a55-a62);
  a59=(a59/a60);
  a62=(a55*a59);
  a61=(a61+a62);
  a62=(a52*a20);
  a56=(a56-a62);
  a33=(a33/a60);
  a62=(a56*a33);
  a61=(a61+a62);
  a62=(a61*a24);
  a40=(a40-a62);
  a62=(a14*a0);
  a63=(a14*a32);
  a62=(a62+a63);
  a63=(a38*a34);
  a62=(a62+a63);
  a63=(a17*a37);
  a62=(a62+a63);
  a63=(a21*a35);
  a62=(a62+a63);
  a63=(a47*a23);
  a62=(a62+a63);
  a63=(a62*a0);
  a63=(a14-a63);
  a64=(a63*a10);
  a65=(a62*a32);
  a14=(a14-a65);
  a65=(a14*a1);
  a64=(a64+a65);
  a65=(a62*a34);
  a38=(a38-a65);
  a65=(a38*a13);
  a64=(a64+a65);
  a65=(a62*a37);
  a17=(a17-a65);
  a65=(a17*a44);
  a64=(a64+a65);
  a65=(a62*a35);
  a65=(a21-a65);
  a66=(a65*a46);
  a64=(a64+a66);
  a66=(a62*a23);
  a47=(a47-a66);
  a66=(a47*a20);
  a64=(a64+a66);
  a66=(a64*a10);
  a63=(a63-a66);
  a66=(a63*a24);
  a67=(a64*a1);
  a14=(a14-a67);
  a67=(a14*a18);
  a66=(a66+a67);
  a67=(a64*a13);
  a38=(a38-a67);
  a67=(a38*a12);
  a66=(a66+a67);
  a67=(a64*a44);
  a17=(a17-a67);
  a67=(a17*a43);
  a66=(a66+a67);
  a67=(a64*a46);
  a65=(a65-a67);
  a67=(a65*a59);
  a66=(a66+a67);
  a67=(a64*a20);
  a47=(a47-a67);
  a67=(a47*a33);
  a66=(a66+a67);
  a67=(a66*a24);
  a63=(a63-a67);
  a67=casadi_sq(a63);
  a68=(a66*a18);
  a14=(a14-a68);
  a68=casadi_sq(a14);
  a67=(a67+a68);
  a68=(a66*a12);
  a38=(a38-a68);
  a68=casadi_sq(a38);
  a67=(a67+a68);
  a68=(a66*a43);
  a17=(a17-a68);
  a68=casadi_sq(a17);
  a67=(a67+a68);
  a68=(a66*a59);
  a65=(a65-a68);
  a68=casadi_sq(a65);
  a67=(a67+a68);
  a68=(a66*a33);
  a47=(a47-a68);
  a68=casadi_sq(a47);
  a67=(a67+a68);
  a67=sqrt(a67);
  a63=(a63/a67);
  a68=(a40*a63);
  a69=(a61*a18);
  a39=(a39-a69);
  a14=(a14/a67);
  a69=(a39*a14);
  a68=(a68+a69);
  a69=(a61*a12);
  a53=(a53-a69);
  a38=(a38/a67);
  a69=(a53*a38);
  a68=(a68+a69);
  a69=(a61*a43);
  a54=(a54-a69);
  a17=(a17/a67);
  a69=(a54*a17);
  a68=(a68+a69);
  a69=(a61*a59);
  a55=(a55-a69);
  a65=(a65/a67);
  a69=(a55*a65);
  a68=(a68+a69);
  a69=(a61*a33);
  a56=(a56-a69);
  a47=(a47/a67);
  a69=(a56*a47);
  a68=(a68+a69);
  a69=(a68*a63);
  a40=(a40-a69);
  a69=(a50*a0);
  a70=(a50*a32);
  a69=(a69+a70);
  a70=(a19*a34);
  a69=(a69+a70);
  a70=(a21*a37);
  a69=(a69+a70);
  a70=(a9*a35);
  a69=(a69+a70);
  a70=(a11*a23);
  a69=(a69+a70);
  a70=(a69*a0);
  a70=(a50-a70);
  a71=(a70*a10);
  a72=(a69*a32);
  a50=(a50-a72);
  a72=(a50*a1);
  a71=(a71+a72);
  a72=(a69*a34);
  a19=(a19-a72);
  a72=(a19*a13);
  a71=(a71+a72);
  a72=(a69*a37);
  a21=(a21-a72);
  a72=(a21*a44);
  a71=(a71+a72);
  a72=(a69*a35);
  a9=(a9-a72);
  a72=(a9*a46);
  a71=(a71+a72);
  a72=(a69*a23);
  a11=(a11-a72);
  a72=(a11*a20);
  a71=(a71+a72);
  a72=(a71*a10);
  a70=(a70-a72);
  a72=(a70*a24);
  a73=(a71*a1);
  a50=(a50-a73);
  a73=(a50*a18);
  a72=(a72+a73);
  a73=(a71*a13);
  a19=(a19-a73);
  a73=(a19*a12);
  a72=(a72+a73);
  a73=(a71*a44);
  a21=(a21-a73);
  a73=(a21*a43);
  a72=(a72+a73);
  a73=(a71*a46);
  a9=(a9-a73);
  a73=(a9*a59);
  a72=(a72+a73);
  a73=(a71*a20);
  a11=(a11-a73);
  a73=(a11*a33);
  a72=(a72+a73);
  a73=(a72*a24);
  a70=(a70-a73);
  a73=(a70*a63);
  a74=(a72*a18);
  a50=(a50-a74);
  a74=(a50*a14);
  a73=(a73+a74);
  a74=(a72*a12);
  a19=(a19-a74);
  a74=(a19*a38);
  a73=(a73+a74);
  a74=(a72*a43);
  a21=(a21-a74);
  a74=(a21*a17);
  a73=(a73+a74);
  a74=(a72*a59);
  a9=(a9-a74);
  a74=(a9*a65);
  a73=(a73+a74);
  a74=(a72*a33);
  a11=(a11-a74);
  a74=(a11*a47);
  a73=(a73+a74);
  a74=(a73*a63);
  a70=(a70-a74);
  a74=casadi_sq(a70);
  a75=(a73*a14);
  a50=(a50-a75);
  a75=casadi_sq(a50);
  a74=(a74+a75);
  a75=(a73*a38);
  a19=(a19-a75);
  a75=casadi_sq(a19);
  a74=(a74+a75);
  a75=(a73*a17);
  a21=(a21-a75);
  a75=casadi_sq(a21);
  a74=(a74+a75);
  a75=(a73*a65);
  a9=(a9-a75);
  a75=casadi_sq(a9);
  a74=(a74+a75);
  a75=(a73*a47);
  a11=(a11-a75);
  a75=casadi_sq(a11);
  a74=(a74+a75);
  a74=sqrt(a74);
  a70=(a70/a74);
  a75=(a40*a70);
  a76=(a68*a14);
  a39=(a39-a76);
  a50=(a50/a74);
  a76=(a39*a50);
  a75=(a75+a76);
  a76=(a68*a38);
  a53=(a53-a76);
  a19=(a19/a74);
  a76=(a53*a19);
  a75=(a75+a76);
  a76=(a68*a17);
  a54=(a54-a76);
  a21=(a21/a74);
  a76=(a54*a21);
  a75=(a75+a76);
  a76=(a68*a65);
  a55=(a55-a76);
  a9=(a9/a74);
  a76=(a55*a9);
  a75=(a75+a76);
  a76=(a68*a47);
  a56=(a56-a76);
  a11=(a11/a74);
  a76=(a56*a11);
  a75=(a75+a76);
  a76=(a75*a70);
  a40=(a40-a76);
  a76=casadi_sq(a40);
  a77=(a75*a50);
  a39=(a39-a77);
  a77=casadi_sq(a39);
  a76=(a76+a77);
  a77=(a75*a19);
  a53=(a53-a77);
  a77=casadi_sq(a53);
  a76=(a76+a77);
  a77=(a75*a21);
  a54=(a54-a77);
  a77=casadi_sq(a54);
  a76=(a76+a77);
  a77=(a75*a9);
  a55=(a55-a77);
  a77=casadi_sq(a55);
  a76=(a76+a77);
  a77=(a75*a11);
  a56=(a56-a77);
  a77=casadi_sq(a56);
  a76=(a76+a77);
  a76=sqrt(a76);
  a40=(a40/a76);
  a40=(a40/a76);
  a77=(a29*a40);
  a0=(a0-a77);
  a77=(a75*a40);
  a70=(a70-a77);
  a70=(a70/a74);
  a77=(a69*a70);
  a0=(a0-a77);
  a77=(a68*a40);
  a63=(a63-a77);
  a77=(a73*a70);
  a63=(a63-a77);
  a63=(a63/a67);
  a77=(a62*a63);
  a0=(a0-a77);
  a77=(a61*a40);
  a24=(a24-a77);
  a77=(a72*a70);
  a24=(a24-a77);
  a77=(a66*a63);
  a24=(a24-a77);
  a24=(a24/a60);
  a77=(a57*a24);
  a0=(a0-a77);
  a77=(a52*a40);
  a10=(a10-a77);
  a77=(a71*a70);
  a10=(a10-a77);
  a77=(a64*a63);
  a10=(a10-a77);
  a77=(a36*a24);
  a10=(a10-a77);
  a10=(a10/a45);
  a77=(a49*a10);
  a0=(a0-a77);
  a0=(a0/a22);
  a77=arg[2] ? arg[2][0] : 0;
  a78=arg[1] ? arg[1][2] : 0;
  a79=arg[1] ? arg[1][0] : 0;
  a80=arg[1] ? arg[1][1] : 0;
  a79=(a79+a80);
  a80=(a2*a79);
  a81=(a41*a80);
  a82=(a78*a81);
  a58=(a58*a78);
  a83=(a80*a58);
  a82=(a82-a83);
  a83=(a80*a78);
  a84=(a4*a83);
  a82=(a82-a84);
  a84=(a26*a78);
  a85=(a27*a80);
  a84=(a84+a85);
  a85=(a26*a80);
  a86=(a27*a78);
  a85=(a85-a86);
  a86=(a51*a85);
  a87=(a84*a86);
  a88=(a5*a84);
  a89=(a85*a88);
  a87=(a87-a89);
  a79=(a3*a79);
  a89=(a30*a79);
  a90=(a28*a79);
  a91=(a31*a90);
  a92=(a89*a91);
  a87=(a87-a92);
  a92=(a31*a89);
  a93=(a90*a92);
  a87=(a87+a93);
  a93=(a5*a83);
  a87=(a87-a93);
  a93=(a15*a84);
  a94=(a16*a85);
  a93=(a93+a94);
  a94=(a15*a85);
  a95=(a16*a84);
  a94=(a94-a95);
  a95=(a48*a94);
  a96=arg[1] ? arg[1][3] : 0;
  a97=(a79+a96);
  a98=(a16*a97);
  a99=(a15*a90);
  a98=(a98-a99);
  a99=(a16*a89);
  a98=(a98-a99);
  a99=(a31*a98);
  a95=(a95+a99);
  a99=(a93*a95);
  a100=(a42*a93);
  a101=-1.2190000000000001e+00;
  a102=(a15*a97);
  a103=(a16*a90);
  a102=(a102+a103);
  a103=(a15*a89);
  a102=(a102-a103);
  a103=(a101*a102);
  a100=(a100+a103);
  a103=(a94*a100);
  a99=(a99-a103);
  a103=(a31*a94);
  a104=(a31*a98);
  a103=(a103+a104);
  a104=(a102*a103);
  a99=(a99-a104);
  a104=(a101*a93);
  a105=(a31*a102);
  a104=(a104+a105);
  a105=(a98*a104);
  a99=(a99+a105);
  a105=(a5*a83);
  a99=(a99-a105);
  a105=(a30*a83);
  a106=arg[0] ? arg[0][1] : 0;
  a107=cos(a106);
  a108=9.8100000000000005e+00;
  a109=arg[0] ? arg[0][0] : 0;
  a110=sin(a109);
  a110=(a108*a110);
  a111=(a107*a110);
  a106=sin(a106);
  a109=cos(a109);
  a108=(a108*a109);
  a109=(a106*a108);
  a111=(a111+a109);
  a109=(a27*a111);
  a105=(a105-a109);
  a107=(a107*a108);
  a106=(a106*a110);
  a107=(a107-a106);
  a106=(a3*a107);
  a110=(a26*a106);
  a105=(a105+a110);
  a110=(a90*a96);
  a105=(a105+a110);
  a110=(a15*a105);
  a108=(a15*a83);
  a109=(a28*a83);
  a112=(a26*a111);
  a109=(a109+a112);
  a112=(a27*a106);
  a109=(a109+a112);
  a112=(a89*a96);
  a109=(a109-a112);
  a112=(a16*a109);
  a108=(a108+a112);
  a110=(a110-a108);
  a108=arg[1] ? arg[1][4] : 0;
  a112=(a98*a108);
  a110=(a110-a112);
  a112=(a7*a110);
  a113=(a7*a83);
  a114=(a15*a109);
  a115=(a16*a83);
  a114=(a114-a115);
  a115=(a16*a105);
  a114=(a114+a115);
  a115=(a102*a108);
  a114=(a114+a115);
  a115=(a8*a114);
  a113=(a113+a115);
  a112=(a112-a113);
  a113=(a97+a108);
  a115=(a8*a113);
  a116=(a7*a98);
  a115=(a115+a116);
  a116=(a8*a102);
  a115=(a115+a116);
  a116=arg[1] ? arg[1][5] : 0;
  a117=(a115*a116);
  a112=(a112-a117);
  a117=(a6*a83);
  a118=(a79*a78);
  a119=(a27*a118);
  a120=(a85*a96);
  a119=(a119+a120);
  a120=(a15*a119);
  a121=(a26*a118);
  a96=(a84*a96);
  a121=(a121-a96);
  a96=(a16*a121);
  a120=(a120+a96);
  a96=(a94*a108);
  a120=(a120+a96);
  a96=(a7*a120);
  a122=(a15*a121);
  a123=(a16*a119);
  a122=(a122-a123);
  a108=(a93*a108);
  a122=(a122-a108);
  a108=(a8*a122);
  a96=(a96+a108);
  a108=(a7*a94);
  a123=(a8*a93);
  a108=(a108-a123);
  a123=(a108*a116);
  a96=(a96+a123);
  a117=(a117+a96);
  a123=(a7*a122);
  a124=(a8*a120);
  a123=(a123-a124);
  a124=(a7*a93);
  a125=(a8*a94);
  a124=(a124+a125);
  a125=(a124*a116);
  a123=(a123-a125);
  a117=(a117+a123);
  a125=(a7*a114);
  a126=(a8*a83);
  a125=(a125-a126);
  a126=(a8*a110);
  a125=(a125+a126);
  a126=(a7*a113);
  a127=(a8*a98);
  a126=(a126-a127);
  a127=(a7*a102);
  a126=(a126+a127);
  a127=(a126*a116);
  a125=(a125+a127);
  a117=(a117+a125);
  a117=(a112-a117);
  a127=(a6*a108);
  a116=(a113+a116);
  a128=(a116+a124);
  a127=(a127-a128);
  a128=(a25*a80);
  a129=(a128-a85);
  a130=(a129-a94);
  a127=(a127-a130);
  a127=(a127+a115);
  a131=(a124*a127);
  a132=(a6*a124);
  a132=(a132-a116);
  a132=(a132-a108);
  a132=(a132+a130);
  a132=(a132-a126);
  a133=(a108*a132);
  a131=(a131-a133);
  a133=(a108-a116);
  a133=(a133+a115);
  a134=(a126*a133);
  a131=(a131-a134);
  a134=(a116-a124);
  a134=(a134+a126);
  a135=(a115*a134);
  a131=(a131+a135);
  a117=(a117+a131);
  a131=(a83+a123);
  a131=(a131+a125);
  a135=(a124-a108);
  a135=(a135+a130);
  a136=(a108*a135);
  a137=(a116*a134);
  a136=(a136-a137);
  a131=(a131+a136);
  a136=(a8*a131);
  a136=(a117+a136);
  a137=(a83+a96);
  a137=(a112-a137);
  a138=(a116*a133);
  a139=(a124*a135);
  a138=(a138-a139);
  a137=(a137+a138);
  a138=(a7*a137);
  a136=(a136+a138);
  a99=(a99+a136);
  a136=(a31*a122);
  a138=(a31*a114);
  a136=(a136+a138);
  a138=(a31*a129);
  a139=(a94*a138);
  a140=(a113*a104);
  a139=(a139-a140);
  a136=(a136+a139);
  a139=(a7*a131);
  a140=(a8*a137);
  a139=(a139-a140);
  a136=(a136+a139);
  a139=(a16*a136);
  a139=(a99+a139);
  a140=(a101*a120);
  a141=(a31*a110);
  a140=(a140+a141);
  a141=(a113*a103);
  a142=(a93*a138);
  a141=(a141-a142);
  a140=(a140+a141);
  a131=(a8*a131);
  a137=(a7*a137);
  a131=(a131+a137);
  a140=(a140+a131);
  a131=(a15*a140);
  a139=(a139+a131);
  a87=(a87+a139);
  a109=(a31*a109);
  a139=(a31*a128);
  a131=(a85*a139);
  a137=(a97*a92);
  a131=(a131+a137);
  a109=(a109+a131);
  a131=(a15*a136);
  a137=(a16*a140);
  a131=(a131-a137);
  a109=(a109+a131);
  a28=(a28*a109);
  a28=(a87-a28);
  a105=(a31*a105);
  a109=(a84*a139);
  a131=(a97*a91);
  a109=(a109+a131);
  a105=(a105-a109);
  a136=(a16*a136);
  a140=(a15*a140);
  a136=(a136+a140);
  a105=(a105+a136);
  a30=(a30*a105);
  a28=(a28-a30);
  a82=(a82+a28);
  a3=(a3*a82);
  a41=(a41*a118);
  a82=2.2750000000000001e-01;
  a82=(a82*a111);
  a41=(a41+a82);
  a58=(a79*a58);
  a4=(a4*a79);
  a78=(a78*a4);
  a58=(a58-a78);
  a41=(a41+a58);
  a119=(a5*a119);
  a58=(a5*a97);
  a78=(a85*a58);
  a86=(a97*a86);
  a78=(a78-a86);
  a89=(a89*a139);
  a78=(a78-a89);
  a89=(a128*a92);
  a78=(a78+a89);
  a119=(a119+a78);
  a42=(a42*a120);
  a101=(a101*a110);
  a42=(a42+a101);
  a5=(a5*a113);
  a101=(a94*a5);
  a95=(a113*a95);
  a101=(a101-a95);
  a102=(a102*a138);
  a101=(a101+a102);
  a102=(a129*a104);
  a101=(a101-a102);
  a42=(a42+a101);
  a101=(a6*a96);
  a101=(a83+a101);
  a101=(a101-a123);
  a118=(a25*a118);
  a107=(a2*a107);
  a118=(a118-a107);
  a107=(a118-a121);
  a102=(a107-a122);
  a101=(a101+a102);
  a101=(a101-a112);
  a112=(a6*a116);
  a112=(a112-a124);
  a112=(a112-a108);
  a112=(a112-a115);
  a112=(a112+a126);
  a95=(a108*a112);
  a127=(a116*a127);
  a95=(a95-a127);
  a126=(a126*a135);
  a95=(a95+a126);
  a126=(a130*a134);
  a95=(a95-a126);
  a101=(a101+a95);
  a95=(a7*a101);
  a83=(a83-a96);
  a6=(a6*a123);
  a83=(a83+a6);
  a83=(a83-a102);
  a83=(a83+a125);
  a116=(a116*a132);
  a112=(a124*a112);
  a116=(a116-a112);
  a115=(a115*a135);
  a116=(a116-a115);
  a130=(a130*a133);
  a116=(a116+a130);
  a83=(a83+a116);
  a116=(a8*a83);
  a95=(a95-a116);
  a42=(a42+a95);
  a95=(a15*a42);
  a48=(a48*a122);
  a114=(a31*a114);
  a48=(a48+a114);
  a113=(a113*a100);
  a5=(a93*a5);
  a113=(a113-a5);
  a98=(a98*a138);
  a113=(a113-a98);
  a129=(a129*a103);
  a113=(a113+a129);
  a48=(a48+a113);
  a8=(a8*a101);
  a7=(a7*a83);
  a8=(a8+a7);
  a96=(a96-a123);
  a96=(a96+a102);
  a124=(a124*a134);
  a108=(a108*a133);
  a124=(a124-a108);
  a96=(a96+a124);
  a8=(a8-a96);
  a48=(a48+a8);
  a8=(a16*a48);
  a95=(a95-a8);
  a119=(a119+a95);
  a95=(a27*a119);
  a51=(a51*a121);
  a97=(a97*a88);
  a58=(a84*a58);
  a97=(a97-a58);
  a90=(a90*a139);
  a97=(a97+a90);
  a128=(a128*a91);
  a97=(a97-a128);
  a51=(a51+a97);
  a16=(a16*a42);
  a15=(a15*a48);
  a16=(a16+a15);
  a107=(a31*a107);
  a93=(a93*a104);
  a94=(a94*a103);
  a93=(a93-a94);
  a107=(a107+a93);
  a107=(a107+a96);
  a16=(a16-a107);
  a51=(a51+a16);
  a16=(a26*a51);
  a95=(a95+a16);
  a31=(a31*a118);
  a85=(a85*a91);
  a84=(a84*a92);
  a85=(a85-a84);
  a31=(a31+a85);
  a31=(a31+a107);
  a25=(a25*a31);
  a95=(a95+a25);
  a41=(a41+a95);
  a2=(a2*a41);
  a3=(a3+a2);
  a77=(a77-a3);
  a0=(a0*a77);
  a39=(a39/a76);
  a39=(a39/a76);
  a2=(a29*a39);
  a32=(a32-a2);
  a2=(a75*a39);
  a50=(a50-a2);
  a50=(a50/a74);
  a2=(a69*a50);
  a32=(a32-a2);
  a2=(a68*a39);
  a14=(a14-a2);
  a2=(a73*a50);
  a14=(a14-a2);
  a14=(a14/a67);
  a2=(a62*a14);
  a32=(a32-a2);
  a2=(a61*a39);
  a18=(a18-a2);
  a2=(a72*a50);
  a18=(a18-a2);
  a2=(a66*a14);
  a18=(a18-a2);
  a18=(a18/a60);
  a2=(a57*a18);
  a32=(a32-a2);
  a2=(a52*a39);
  a1=(a1-a2);
  a2=(a71*a50);
  a1=(a1-a2);
  a2=(a64*a14);
  a1=(a1-a2);
  a2=(a36*a18);
  a1=(a1-a2);
  a1=(a1/a45);
  a2=(a49*a1);
  a32=(a32-a2);
  a32=(a32/a22);
  a2=arg[2] ? arg[2][1] : 0;
  a2=(a2-a3);
  a32=(a32*a2);
  a0=(a0+a32);
  a53=(a53/a76);
  a53=(a53/a76);
  a32=(a29*a53);
  a34=(a34-a32);
  a32=(a75*a53);
  a19=(a19-a32);
  a19=(a19/a74);
  a32=(a69*a19);
  a34=(a34-a32);
  a32=(a68*a53);
  a38=(a38-a32);
  a32=(a73*a19);
  a38=(a38-a32);
  a38=(a38/a67);
  a32=(a62*a38);
  a34=(a34-a32);
  a32=(a61*a53);
  a12=(a12-a32);
  a32=(a72*a19);
  a12=(a12-a32);
  a32=(a66*a38);
  a12=(a12-a32);
  a12=(a12/a60);
  a32=(a57*a12);
  a34=(a34-a32);
  a32=(a52*a53);
  a13=(a13-a32);
  a32=(a71*a19);
  a13=(a13-a32);
  a32=(a64*a38);
  a13=(a13-a32);
  a32=(a36*a12);
  a13=(a13-a32);
  a13=(a13/a45);
  a32=(a49*a13);
  a34=(a34-a32);
  a34=(a34/a22);
  a32=arg[2] ? arg[2][2] : 0;
  a3=-2.2750000000000001e-01;
  a3=(a3*a106);
  a80=(a80*a4);
  a79=(a79*a81);
  a80=(a80-a79);
  a3=(a3+a80);
  a26=(a26*a119);
  a27=(a27*a51);
  a26=(a26-a27);
  a3=(a3+a26);
  a32=(a32-a3);
  a34=(a34*a32);
  a0=(a0+a34);
  a54=(a54/a76);
  a54=(a54/a76);
  a34=(a29*a54);
  a37=(a37-a34);
  a34=(a75*a54);
  a21=(a21-a34);
  a21=(a21/a74);
  a34=(a69*a21);
  a37=(a37-a34);
  a34=(a68*a54);
  a17=(a17-a34);
  a34=(a73*a21);
  a17=(a17-a34);
  a17=(a17/a67);
  a34=(a62*a17);
  a37=(a37-a34);
  a34=(a61*a54);
  a43=(a43-a34);
  a34=(a72*a21);
  a43=(a43-a34);
  a34=(a66*a17);
  a43=(a43-a34);
  a43=(a43/a60);
  a34=(a57*a43);
  a37=(a37-a34);
  a34=(a52*a54);
  a44=(a44-a34);
  a34=(a71*a21);
  a44=(a44-a34);
  a34=(a64*a17);
  a44=(a44-a34);
  a34=(a36*a43);
  a44=(a44-a34);
  a44=(a44/a45);
  a34=(a49*a44);
  a37=(a37-a34);
  a37=(a37/a22);
  a34=arg[2] ? arg[2][3] : 0;
  a34=(a34-a87);
  a37=(a37*a34);
  a0=(a0+a37);
  a55=(a55/a76);
  a55=(a55/a76);
  a37=(a29*a55);
  a35=(a35-a37);
  a37=(a75*a55);
  a9=(a9-a37);
  a9=(a9/a74);
  a37=(a69*a9);
  a35=(a35-a37);
  a37=(a68*a55);
  a65=(a65-a37);
  a37=(a73*a9);
  a65=(a65-a37);
  a65=(a65/a67);
  a37=(a62*a65);
  a35=(a35-a37);
  a37=(a61*a55);
  a59=(a59-a37);
  a37=(a72*a9);
  a59=(a59-a37);
  a37=(a66*a65);
  a59=(a59-a37);
  a59=(a59/a60);
  a37=(a57*a59);
  a35=(a35-a37);
  a37=(a52*a55);
  a46=(a46-a37);
  a37=(a71*a9);
  a46=(a46-a37);
  a37=(a64*a65);
  a46=(a46-a37);
  a37=(a36*a59);
  a46=(a46-a37);
  a46=(a46/a45);
  a37=(a49*a46);
  a35=(a35-a37);
  a35=(a35/a22);
  a37=arg[2] ? arg[2][4] : 0;
  a37=(a37-a99);
  a35=(a35*a37);
  a0=(a0+a35);
  a56=(a56/a76);
  a56=(a56/a76);
  a29=(a29*a56);
  a23=(a23-a29);
  a75=(a75*a56);
  a11=(a11-a75);
  a11=(a11/a74);
  a69=(a69*a11);
  a23=(a23-a69);
  a68=(a68*a56);
  a47=(a47-a68);
  a73=(a73*a11);
  a47=(a47-a73);
  a47=(a47/a67);
  a62=(a62*a47);
  a23=(a23-a62);
  a61=(a61*a56);
  a33=(a33-a61);
  a72=(a72*a11);
  a33=(a33-a72);
  a66=(a66*a47);
  a33=(a33-a66);
  a33=(a33/a60);
  a57=(a57*a33);
  a23=(a23-a57);
  a52=(a52*a56);
  a20=(a20-a52);
  a71=(a71*a11);
  a20=(a20-a71);
  a64=(a64*a47);
  a20=(a20-a64);
  a36=(a36*a33);
  a20=(a20-a36);
  a20=(a20/a45);
  a49=(a49*a20);
  a23=(a23-a49);
  a23=(a23/a22);
  a22=arg[2] ? arg[2][5] : 0;
  a22=(a22-a117);
  a23=(a23*a22);
  a0=(a0+a23);
  if (res[0]!=0) res[0][0]=a0;
  a10=(a10*a77);
  a1=(a1*a2);
  a10=(a10+a1);
  a13=(a13*a32);
  a10=(a10+a13);
  a44=(a44*a34);
  a10=(a10+a44);
  a46=(a46*a37);
  a10=(a10+a46);
  a20=(a20*a22);
  a10=(a10+a20);
  if (res[0]!=0) res[0][1]=a10;
  a24=(a24*a77);
  a18=(a18*a2);
  a24=(a24+a18);
  a12=(a12*a32);
  a24=(a24+a12);
  a43=(a43*a34);
  a24=(a24+a43);
  a59=(a59*a37);
  a24=(a24+a59);
  a33=(a33*a22);
  a24=(a24+a33);
  if (res[0]!=0) res[0][2]=a24;
  a63=(a63*a77);
  a14=(a14*a2);
  a63=(a63+a14);
  a38=(a38*a32);
  a63=(a63+a38);
  a17=(a17*a34);
  a63=(a63+a17);
  a65=(a65*a37);
  a63=(a63+a65);
  a47=(a47*a22);
  a63=(a63+a47);
  if (res[0]!=0) res[0][3]=a63;
  a70=(a70*a77);
  a50=(a50*a2);
  a70=(a70+a50);
  a19=(a19*a32);
  a70=(a70+a19);
  a21=(a21*a34);
  a70=(a70+a21);
  a9=(a9*a37);
  a70=(a70+a9);
  a11=(a11*a22);
  a70=(a70+a11);
  if (res[0]!=0) res[0][4]=a70;
  a40=(a40*a77);
  a39=(a39*a2);
  a40=(a40+a39);
  a53=(a53*a32);
  a40=(a40+a53);
  a54=(a54*a34);
  a40=(a40+a54);
  a55=(a55*a37);
  a40=(a40+a55);
  a56=(a56*a22);
  a40=(a40+a56);
  if (res[0]!=0) res[0][5]=a40;
  return 0;
}

CASADI_SYMBOL_EXPORT int q_ddot(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void q_ddot_incref(void) {
}

CASADI_SYMBOL_EXPORT void q_ddot_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int q_ddot_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int q_ddot_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT const char* q_ddot_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* q_ddot_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* q_ddot_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* q_ddot_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int q_ddot_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif