/*
Solution of Stationary Advection Diffusion problem with interior singularity on a
square domain using THB adaptive refinement.

This file is part of my Master Thesis.

Author : Argyrios Ntoumanis
*/

# include <gismo.h>
# include <gsAssembler/gsAdaptiveRefUtils.h>

using namespace gismo;

int main(int argc, char* argv[])
{
    // --------------- Initialize Paraview ---------------
    bool plot = true;
    gsCmdLine cmd("Example for solving a convection-diffusion problem.");
    cmd.addSwitch("plot", "Create a ParaView visualization file with the solution", plot);
    try { cmd.getValues(argc, argv); }
    catch (int rv) { return rv; }

    // --------------- specify exact solution and right-hand-side ---------------
    gsFunctionExpr<> g("if( y>=0.7, if(x!=1, 1, 0 ), 0)", 2);
    gsFunctionExpr<> rhs("0", 2); // Define source function
    gsFunctionExpr<> coeff_diff("0.00000001", "0", "0", "0.00000001", 2); // diffusion coefficient
    gsFunctionExpr<> coeff_conv("cos(-pi/3)", "sin(-pi/3)", 2);
    gsFunctionExpr<> coeff_reac("0", 2); // reaction coefficient
    gsInfo << "Source function " << rhs << "\n";
    gsInfo << "Dirichlet boundary conditions " << g << "\n\n";

    // --------------- read geometry from file ---------------
    gsMultiPatch<> mp(*gsNurbsCreator<>::BSplineSquare(1, 0.0, 0.0));
    mp.computeTopology();
    gsTensorBSpline<2, real_t>* geo = dynamic_cast<gsTensorBSpline<2, real_t> *>(&mp.patch(0));
    gsTensorBSplineBasis<2, real_t> tbb = geo->basis();
    gsTHBSplineBasis<2, real_t> THB(tbb);
    gsTHBSpline<2, real_t> THB_patches(tbb, geo->coefs());
    gsMultiBasis<real_t> bases(THB);
    gsMultiPatch<real_t> patches(THB_patches);
    patches.computeTopology();

    // --------------- add boundary conditions ---------------
    gsBoundaryConditions<> bcInfo;
    for (gsMultiPatch<>::const_biterator
        bit = patches.bBegin(); bit != patches.bEnd(); ++bit)
    {
        bcInfo.addCondition(*bit, condition_type::dirichlet, &g);
    }    

    // ------------ apply initial uniform refinement --------------  
    //bases.degreeElevate(); //p=2
    //bases.degreeElevate(); //p=3
    //bases.degreeElevate(); //p=4
    int numInitUniformRefine = 2;//8
    for (int i = 0; i < numInitUniformRefine; ++i)
        bases.uniformRefine();
    gsInfo << "Size of THB basis is: " << bases.basis(0).size() << "\n\n";
    gsInfo << "Degree of THB basis is " << bases.basis(0).degree(0) << "\n\n";
    
    // --------------- define Pde ---------------
    gsConvDiffRePde<real_t> cdrPde(patches, bcInfo, &coeff_diff, &coeff_conv, &coeff_reac, &rhs);
    gsCDRAssembler<real_t> cdrAss(cdrPde, bases);
    cdrAss.options().setInt("Stabilization", stabilizerCDR::SUPG);
    cdrAss.options().setInt("DirichletValues", dirichlet::l2Projection);

    // --------------- set up adaptive refinement loop ---------------
    int numRefinementLoops = 6;// 0
    MarkingStrategy adaptRefCrit = BULK;
    const real_t adaptRefParam = 0.01;

    // --------------- adaptive refinement loop ---------------
    for (int refLoop = 0; refLoop <= numRefinementLoops; refLoop++)
    {
        gsInfo << "====== Loop " << refLoop << " of "
            << numRefinementLoops << " ======" << "\n";

        // --------------- solving ---------------
        cdrAss.assemble();
        gsSparseSolver<>::BiCGSTABILUT solver(cdrAss.matrix());
        gsMatrix<> solVector = solver.solve(cdrAss.rhs());
        gsField<> solField;
        solField = cdrAss.constructSolution(solVector);

        // --------------- error computation ---------------
        gsExprEvaluator<> ev;
        ev.setIntegrationElements(cdrAss.multiBasis());
        gsExprEvaluator<>::geometryMap Gm = ev.getMap(patches);
        gsExprEvaluator<>::variable is = ev.getVariable(solField.fields());
        auto ms = ev.getVariable(g, Gm);
        ev.integralElWise((igrad(is, Gm) - igrad(ms)).sqNorm() * meas(Gm));

        gsInfo << "System matrix nonzero elements: " << cdrAss.matrix().nonZeros();

        const std::vector<real_t>& eltErrs = ev.elementwise();

        // --------------- adaptive refinement ---------------
        std::vector<bool> elMarked(eltErrs.size());
        gsMarkElementsForRef(eltErrs, adaptRefCrit, adaptRefParam, elMarked);
        //gsInfo << "Marked " << std::count(elMarked.begin(), elMarked.end(), true) << " elements.\n\n";
        if (plot && refLoop == numRefinementLoops)
        {
            gsWriteParaview<>(solField, "DiffusionInteriorSquare_mesh", 1000, true);
            gsWriteParaview<>(solField, "DiffusionInteriorSquare_solution", 1000, false);
            //const gsField<> exact(cdrAss.patches(), g, false);
            //gsWriteParaview<>(exact, "poisson2d_exact", 1000, false);
        }
        gsRefineMarkedElements(cdrAss.multiBasis(), elMarked, 1);
        gsRefineMarkedElements(bases, elMarked, 1);
        cdrAss.multiBasis().repairInterfaces(patches.interfaces());
        bases.repairInterfaces(patches.interfaces());
        cdrAss.refresh();
        gsInfo << "\nSize of THB basis is: " << bases.basis(0).size() << "\n\n";
    }

    return EXIT_SUCCESS;
}
