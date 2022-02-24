/*
Least square approximation using THB-splines to approximate given data set.

This file is part of my Master Thesis.

Author : Argyrios Ntoumanis
*/

#include <gismo.h>

using namespace gismo;

int main(int argc, char* argv[])
{
    bool plot = false;
    index_t numURef = 3;
    index_t iter = 10;
    index_t deg_x = 2;
    index_t deg_y = 2;

    // 1st Example;
    //real_t lambda = 1e-7;
    //index_t extension = 1;
    //real_t threshold = 1e-03; // 1e-02, 5e-02, 1e-01, -1
    //real_t tolerance = 1e-03; // 1e-02, 5e-02, 1e-01
    //real_t refPercent = 0.1; // 0.3 , 0.4 , 0.6
    //std::string fn = "fitting/deepdrawingC.xml";

    // 2nd Example;
    real_t lambda = 1e-9; // 1e-7, 1e-8, 1e-10
    index_t extension = 2; // 3, 4, 5
    real_t threshold = -1; // 1e-7
    real_t tolerance = 1.8e-6;
    real_t refPercent = 0.1; // 0.05, 0.2
    std::string fn = "fitting/3peaksdrawing.xml";

    gsCmdLine cmd("Fit parametrized sample data.");
    cmd.addSwitch("plot", "Plot the result in ParaView.", plot);
    try { cmd.getValues(argc, argv); }
    catch (int rv) { return rv; }
    if (extension < 0)
    {
        gsInfo << "Extension must be non negative.\n"; return 0;
    }
    if (tolerance < 0)
    {
        gsInfo << "Error tolerance cannot be negative, setting it to default value.\n";
        tolerance = 1e-02;
    }
    if (threshold > 0 && threshold > tolerance)
    {
        gsInfo << "Refinement threshold is over tolerance, setting it the same as tolerance.\n";
        threshold = tolerance;
    }

    gsFileData<> fd_in(fn);
    gsMatrix<> uv, xyz;
    fd_in.getId<gsMatrix<> >(0, uv);
    fd_in.getId<gsMatrix<> >(1, xyz);
    gsFileData<> fd;
    GISMO_ASSERT(uv.cols() == xyz.cols() && uv.rows() == 2 && xyz.rows() == 3,
        "Wrong input");

    real_t u_min = uv.row(0).minCoeff(),
        u_max = uv.row(0).maxCoeff(),
        v_min = uv.row(1).minCoeff(),
        v_max = uv.row(1).maxCoeff();
    gsInfo << "v_min: " << v_min << "\n" << "v_max: " << v_max << "\n\n";
    gsKnotVector<> u_knots(u_min, u_max, 0, deg_x + 1);
    gsKnotVector<> v_knots(v_min, v_max, 0, deg_y + 1);
    gsTensorBSplineBasis<2> T_tbasis(u_knots, v_knots);
    T_tbasis.uniformRefine((1 << numURef) - 1);
    gsTHBSplineBasis<2>  THB(T_tbasis);

    std::vector<unsigned> ext;
    ext.push_back(extension);
    ext.push_back(extension);

    gsHFitting<2, real_t> ref(uv, xyz, THB, refPercent, ext, lambda);

    const std::vector<real_t>& errors = ref.pointWiseErrors();
    gsInfo << "Fitting " << xyz.cols() << " samples.\n";
    gsInfo << "----------------\n";
    gsInfo << "Cell extension     : " << ext[0] << " " << ext[1] << ".\n";
    if (threshold >= 0.0)
        gsInfo << "Ref. threshold     : " << threshold << ".\n";
    else
        gsInfo << "Cell refinement    : " << 100 * refPercent << "%%.\n";
    gsInfo << "Error tolerance    : " << tolerance << ".\n";
    gsInfo << "Smoothing parameter: " << lambda << ".\n";
    gsStopwatch time;

    for (int i = 0; i <= iter; i++)
    {
        gsInfo << "----------------\n\n";
        gsInfo << "Iteration " << i << ".." << "\n";
        time.restart();
        ref.nextIteration(tolerance, threshold);
        time.stop();
        gsInfo << "Fitting time: " << time << "\n";
        gsInfo << "Fitted with " << ref.result()->basis() << "\n";
        gsInfo << "Min distance : " << ref.minPointError() << " / ";
        gsInfo << "Max distance : " << ref.maxPointError() << "\n";
        gsInfo << "Points below tolerance: " << 100.0 * ref.numPointsBelow(tolerance) / errors.size() << "%.\n";
        if (i == iter)
        {
            gsWriteParaview(ref.result()->basis(), "SurfaceApproximation",1000,true);
        }
        if (ref.maxPointError() < tolerance)
        {
            gsInfo << "Error tolerance achieved after " << i << " iterations.\n";
            gsWriteParaview(ref.result()->basis(), "SurfaceApproximation", 1000, true);
            break;
        }
    }

    return 0;
}