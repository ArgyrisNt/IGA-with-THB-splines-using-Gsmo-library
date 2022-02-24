/*
Solution of Stationary Advection Diffusion problem with two dimensional boundary singularity on a 
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
    gsFunctionExpr<> g("exp((y-1)*37.5/2)*exp((x-1)*37.5/2)", 2); // Define exact solution 
    gsFunctionExpr<> rhs("0", 2); // Define source function
    gsFunctionExpr<> coeff_diff("1/37.5", "0", "0", "1/37.5", 2); // diffusion coefficient
    gsFunctionExpr<> coeff_conv("1", "0", 2); // convection coefficient
    gsFunctionExpr<> coeff_reac("0", 2); // reaction coefficient
    gsInfo << "Source function " << rhs << "\n";
    gsInfo << "Dirichlet boundary conditions " << g << "\n\n";

    // --------------- read geometry from file ---------------
    std::string fileSrc("domain2d/squareTHB.xml");
    gsMultiPatch<real_t> patches;
    gsReadFile<real_t>(fileSrc, patches);
    patches.computeTopology();
    gsMultiBasis<> bases(patches);

    // --------------- add boundary conditions ---------------
    gsBoundaryConditions<> bcInfo;
    for (gsMultiPatch<>::const_biterator
        bit = patches.bBegin(); bit != patches.bEnd(); ++bit)
    {
        bcInfo.addCondition(*bit, condition_type::dirichlet, &g);
    }

    // ------------ apply initial uniform refinement --------------  
    bases.degreeElevate();
    int numInitUniformRefine = 2;//6;
    for (int i = 0; i < numInitUniformRefine; ++i)
        bases.uniformRefine();
    gsInfo << "Size of THB basis is: " << bases.basis(0).size() << "\n\n";
    gsInfo << "Degree of THB basis is: " << bases.basis(0).degree(0) << "\n\n";

    // --------------- define Pde ---------------
    gsConvDiffRePde<real_t> cdrPde(patches, bcInfo, &coeff_diff, &coeff_conv, &coeff_reac, &rhs);
    gsCDRAssembler<real_t> cdrAss(cdrPde, bases);
    cdrAss.options().setInt("Stabilization", stabilizerCDR::SUPG);
    cdrAss.options().setInt("DirichletValues", dirichlet::l2Projection);

    // --------------- set up adaptive refinement loop ---------------
    int numRefinementLoops = 5;//0;
    //MarkingStrategy adaptRefCrit = PUCA;
    //const real_t adaptRefParam = 0.7;
    //MarkingStrategy adaptRefCrit = GARU;
    //const real_t adaptRefParam = 0.05;
    MarkingStrategy adaptRefCrit = BULK;
    const real_t adaptRefParam = 0.05;
    std::vector<real_t> L2s;
    std::vector<gsMatrix<real_t>> solutions;
    gsMatrix<real_t> u(2, 5);
    u(0, 0) = 0.5; u(1, 0) = 0.5; u(0, 1) = 0.75; u(1, 1) = 0.75; u(0, 2) = 0.875;
    u(1, 2) = 0.875; u(0, 3) = 0.75; u(1, 3) = 0.875; u(0, 4) = 0.875; u(1, 4) = 0.75;

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

        // ---------------  error computation ---------------
        gsExprEvaluator<> ev;
        ev.setIntegrationElements(cdrAss.multiBasis());
        gsExprEvaluator<>::geometryMap Gm = ev.getMap(patches);
        gsExprEvaluator<>::variable is = ev.getVariable(solField.fields());
        auto ms = ev.getVariable(g, Gm);
        ev.integralElWise((igrad(is, Gm) - igrad(ms)).sqNorm() * meas(Gm));

        real_t errorL2 = solField.distanceL2(g, false);
        L2s.push_back(errorL2);

        //gsInfo << "Nonzeros entries of Stiffness matrix: " << cdrAss.matrix().nonZeros() << "\n\n";

        const std::vector<real_t>& eltErrs = ev.elementwise();

        // --------------- evaluate approx solution on specific points (on upper right corner) ---------------
        solutions.push_back(solField.value(u, 0));

        // --------------- adaptive refinement ---------------
        std::vector<bool> elMarked(eltErrs.size());
        gsMarkElementsForRef(eltErrs, adaptRefCrit, adaptRefParam, elMarked);
        //gsInfo << "Marked " << std::count(elMarked.begin(), elMarked.end(), true) << " elements.\n";
        if (plot && refLoop == numRefinementLoops)
        {
            gsWriteParaview<>(solField, "DiffusionBoundary2DSquare_mesh", 1000, true);
            gsWriteParaview<>(solField, "DiffusionBoundary2DSquare_solution", 1000, false);

            // --------------- evaluate exact solution on specific points (on x axis) ---------------
            //const gsField<> exact(cdrAss.patches(), g, false);
            //std::vector<gsMatrix<real_t>> exacts;
            //exacts.push_back(exact.value(u, 0));
            //gsWriteParaview<>(exact, "adaptRef_exact", 1000, true);
            gsInfo << "L2 : ";
            for (size_t temp = 0; temp <= numRefinementLoops; temp++)
            {
                gsInfo << std::setprecision(8) << std::setw(15) << L2s[temp] << " | ";
            }
            gsInfo << "\n\n\n";
            gsInfo << std::setw(25) << "u(0.5,0.5)" << std::setw(20) << "u(0.75,0.75)" << std::setw(20) << "u(0.875,0.875)" << std::setw(20)
                << "u(0.75,0.875)" << std::setw(20) << "u(0.875,0.75)" << "\n\n";
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