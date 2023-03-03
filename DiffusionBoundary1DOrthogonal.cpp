/*
Solution of Stationary Advection Diffusion problem with one dimensional boundary singularity on an 
orthogonal domain using THB adaptive refinement.

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
    gsFunctionExpr<> g("cos((pi*y))*exp((x-6.0)*37.5)", 2); // Define exact solution   
    gsFunctionExpr<> rhs("0", 2); // Define source function   
    gsFunctionExpr<> coeff_diff("1/37.5", "0" , "0", "1/37.5", 2); // diffusion coefficient
    gsFunctionExpr<> coeff_conv("1", "0", 2); // convection coefficient
    gsFunctionExpr<> coeff_reac("0", 2); // reaction coefficient
    gsInfo << "Source function " << rhs << "\n";
    gsInfo << "Dirichlet boundary conditions " << g << "\n\n";

    // --------------- read geometry from file ---------------
    gsMultiPatch<> mp(*gsNurbsCreator<>::BSplineRectangle(0.0, 0.0, 6.0, 0.5));
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
    bcInfo.addCondition(boundary::west, condition_type::dirichlet, &rhs);
    bcInfo.addCondition(boundary::east, condition_type::dirichlet, &g);
    bcInfo.addCondition(boundary::north, condition_type::dirichlet, &rhs);
    bcInfo.addCondition(boundary::south, condition_type::neumann, &rhs);

    // ------------ apply initial uniform refinement --------------  
    int numInitUniformRefine = 2; // 7
    for (int i = 0; i < numInitUniformRefine; ++i)
    {
        patches.uniformRefine();
        bases.uniformRefine();
    }
    gsInfo << "Size of THB basis is: " << bases.basis(0).size() << "\n\n";
    
    // --------------- define Pde ---------------
    gsConvDiffRePde<real_t> cdrPde(patches, bcInfo, &coeff_diff, &coeff_conv, &coeff_reac, &rhs);
    gsCDRAssembler<real_t> cdrAss(cdrPde, bases);
    cdrAss.options().setInt("Stabilization", stabilizerCDR::SUPG);
    cdrAss.options().setInt("DirichletValues", dirichlet::l2Projection);

    // --------------- set up adaptive refinement loop ---------------
    int numRefinementLoops = 9;//0 // 12
    //MarkingStrategy adaptRefCrit = PUCA;
    //const real_t adaptRefParam = 0.9;
    //MarkingStrategy adaptRefCrit = GARU;
    //const real_t adaptRefParam = 0.1;
    MarkingStrategy adaptRefCrit = BULK;
    const real_t adaptRefParam = 0.1;
    std::vector<real_t> L2s;
    std::vector<gsMatrix<real_t>> solutions;
    gsMatrix<real_t> u(2, 5);
    u(0, 0) = 0.5; u(1, 0) = 0.0; u(0, 1) = 0.75; u(1, 1) = 0.0; u(0, 2) = 0.88;
    u(1, 2) = 0.0; u(0, 3) = 0.97; u(1, 3) = 0.0; u(0, 4) = 1.0; u(1, 4) = 0.0;

    // --------------- adaptive refinement loop ---------------
    for (int refLoop = 0; refLoop <= numRefinementLoops; refLoop++)
    {
        gsInfo << "====== Loop " << refLoop << " of "
            << numRefinementLoops << " ======" << "\n";

        // --------------- solving ---------------
        cdrAss.assemble();
        gsMatrix<real_t> solVector =
            gsSparseSolver<>::BiCGSTABILUT(cdrAss.matrix()).solve(cdrAss.rhs());
        gsField<> solField;
        solField = cdrAss.constructSolution(solVector);

        // --------------- error computation ---------------
        gsExprEvaluator<> ev;
        ev.setIntegrationElements(cdrAss.multiBasis());
        gsExprEvaluator<>::geometryMap Gm = ev.getMap(patches);
        gsExprEvaluator<>::variable is = ev.getVariable(solField.fields());
        auto ms = ev.getVariable(g, Gm);
        real_t error_energy = ev.integralElWise((igrad(is, Gm) - igrad(ms)).sqNorm() * meas(Gm));

        real_t errorL2 = solField.distanceL2(g, false);
        L2s.push_back(errorL2);

        //gsInfo << "Nonzeros entries of Stiffness matrix: " << cdrAss.matrix().nonZeros() << "\n\n";

        const std::vector<real_t>& eltErrs = ev.elementwise();

        // --------------- evaluate approx solution on specific points (on x axis) ---------------
        solutions.push_back(solField.value(u, 0));

        // --------------- adaptive refinement ---------------
        std::vector<bool> elMarked(eltErrs.size());
        gsMarkElementsForRef(eltErrs, adaptRefCrit, adaptRefParam, elMarked);
        //gsInfo << "Marked " << std::count(elMarked.begin(), elMarked.end(), true) << " elements.\n";
        if (plot && refLoop == numRefinementLoops)
        {
            gsWriteParaview<>(solField, "DiffusionBoundary1DOrthogonal_mesh", 1000, true);
            gsWriteParaview<>(solField, "DiffusionBoundary1DOrthogonal_solution", 1000, false);

            // --------------- evaluate exact solution on specific points (on x axis) ---------------
            //const gsField<> exact(cdrAss.patches(), g, false);
            //gsWriteParaview<>(exact, "DiffusionBoundary1DOrthogonal_exact", 1000, false);
            //std::vector<gsMatrix<real_t>> exacts;
            //exacts.push_back(exact.value(u, 0));   
            gsInfo << "L2 : ";
            for (size_t temp = 0; temp <= numRefinementLoops; temp++)
            {
                gsInfo << std::setprecision(8) << std::setw(15) << L2s[temp] << " | ";
            }
            gsInfo << "\n\n\n";
            gsInfo << std::setw(25) << "u(3.0,0.0)" << std::setw(20) << "u(4.5,0.0)" << std::setw(20) << "u(5.28,0.0)" << std::setw(20) 
                << "u(5.82,0.0)" << std::setw(20) << "u(6.0,0.0)" << "\n\n";
            for (size_t temp = 0; temp < solutions.size(); temp++)
            {
                gsInfo << std::setw(0) << "ref " << temp << std::setprecision(8) << std::setw(20) << solutions[temp][0] << std::setprecision(8) <<
                    std::setw(20) << solutions[temp][1] << std::setprecision(8) << std::setw(20) << solutions[temp][2] 
                    << std::setprecision(8) << std::setw(20) << solutions[temp][3] << std::setprecision(8) << std::setw(20) << solutions[temp][4] << std::endl;
            }
            gsInfo << "\n";
        }
        gsRefineMarkedElements(cdrAss.multiBasis(), elMarked, 1);
        gsRefineMarkedElements(bases, elMarked, 1);
        cdrAss.multiBasis().repairInterfaces(patches.interfaces());
        bases.repairInterfaces(patches.interfaces());
        cdrAss.refresh();
        gsInfo << "Size of THB basis is: " << bases.basis(0).size() << "\n\n";
    }

    return EXIT_SUCCESS;
}