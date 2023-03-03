/*
Solution of Linear Elasticity on plate with hole using THB adaptive refinement.

This file is part of my Master Thesis.

Author : Argyrios Ntoumanis
*/

#include <gismo.h>
#include <gsElasticity/gsElasticityAssembler.h>
#include <gsElasticity/gsWriteParaviewMultiPhysics.h>
#include <gsAssembler/gsAdaptiveRefUtils.h>

using namespace gismo;

int main(int argc, char* argv[]) {

    // --------------- Initialize Paraview ---------------
    bool plotMesh = true;
    gsInfo << "Example for solving a linear elasticity problem.";
    gsCmdLine cmd("This is the 2D linear elasticity benchmark: infinite plate with circular hole.");
    try { cmd.getValues(argc, argv); }
    catch (int rv) { return rv; }

    // --------------- read geometry from file ---------------
    std::string filename = "plateWithHoleMp.xml";
    index_t numUniRef = 3;//6
    index_t numDegElev = 0;//1
    index_t numPlotPoints = 10000;
    real_t youngsModulus = 1.0e3;
    real_t poissonsRatio = 0.3;
    gsMultiPatch<> geometry;
    gsReadFile<>(filename, geometry);
    gsMultiBasis<> basisTens(geometry);
    std::vector< gsBasis<real_t>* > basisContainer;
    for (size_t i = 0; i < basisTens.nBases(); i++)
    {
        basisContainer.push_back(new gsTHBSplineBasis<2, real_t>(basisTens.basis(i)));
    }
    gsMultiBasis<> basis(basisContainer, geometry);

    // ------------ apply degree elevation --------------  
    for (index_t i = 0; i < numDegElev; ++i)
    {
        basis.degreeElevate();
        geometry.degreeElevate();
    }

    // ------------ apply initial uniform refinement --------------  
    for (index_t i = 0; i < numUniRef; ++i)
    {
        basis.uniformRefine();
        geometry.uniformRefine();
    }
    gsInfo << "Size of THB basis is: " << basis.basis(0).size() << "\n\n";
    gsInfo << "Degree of THB basis is " << basis.basis(0).degree(0) << "\n\n";

    // --------------- add boundary conditions and load ---------------
    gsFunctionExpr<> analyticalStresses("1-1/(x^2+y^2)*(3/2*cos(2*atan2(y,x)) + cos(4*atan2(y,x))) + 3/2/(x^2+y^2)^2*cos(4*atan2(y,x))",
        "-1/(x^2+y^2)*(1/2*cos(2*atan2(y,x)) - cos(4*atan2(y,x))) - 3/2/(x^2+y^2)^2*cos(4*atan2(y,x))",
        "-1/(x^2+y^2)*(1/2*sin(2*atan2(y,x)) + sin(4*atan2(y,x))) + 3/2/(x^2+y^2)^2*sin(4*atan2(y,x))", 2);
    gsFunctionExpr<> tractionWest("-1+1/(x^2+y^2)*(3/2*cos(2*atan2(y,x)) + cos(4*atan2(y,x))) - 3/2/(x^2+y^2)^2*cos(4*atan2(y,x))",
        "1/(x^2+y^2)*(1/2*sin(2*atan2(y,x)) + sin(4*atan2(y,x))) - 3/2/(x^2+y^2)^2*sin(4*atan2(y,x))", 2);
    gsFunctionExpr<> tractionNorth("-1/(x^2+y^2)*(1/2*sin(2*atan2(y,x)) + sin(4*atan2(y,x))) + 3/2/(x^2+y^2)^2*sin(4*atan2(y,x))",
        "-1/(x^2+y^2)*(1/2*cos(2*atan2(y,x)) - cos(4*atan2(y,x))) - 3/2/(x^2+y^2)^2*cos(4*atan2(y,x))", 2);
    gsBoundaryConditions<> bcInfo;
    bcInfo.addCondition(0, boundary::north, condition_type::neumann, &tractionWest);
    bcInfo.addCondition(1, boundary::north, condition_type::neumann, &tractionNorth);
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, nullptr, 1); // last number is a component (coordinate) number
    bcInfo.addCondition(1, boundary::east, condition_type::dirichlet, nullptr, 0);
    gsConstantFunction<> g(0., 0., 2); // rhs

    // --------------- set up adaptive refinement loop ---------------
    int numRefinementLoops = 7;//0
    MarkingStrategy adaptRefCrit = PUCA;
    const real_t adaptRefParam = 0.9;

    std::string fileName;
    const std::string baseName("ElasticityPlateWithHoleAdaptiveTHB_mesh");

    // --------------- adaptive refinement loop ---------------
    for (int refLoop = 0; refLoop <= numRefinementLoops; refLoop++)
    {
        gsInfo << "====== Loop " << refLoop << " of "
            << numRefinementLoops << " ======" << "\n";

        // --------------- solving ---------------
        gsElasticityAssembler<real_t> assembler(geometry, basis, bcInfo, g);
        assembler.options().setReal("YoungsModulus", youngsModulus);
        assembler.options().setReal("PoissonsRatio", poissonsRatio);
        gsInfo << "Assembling...\n";
        gsStopwatch clock;
        clock.restart();
        assembler.assemble();
        gsInfo << "Assembled a system (matrix and load vector) with "
            << assembler.numDofs() << " dofs in " << clock.stop() << "s.\n";
        gsInfo << "Solving...\n";
        clock.restart();

#ifdef GISMO_WITH_PARDISO
        gsSparseSolver<>::PardisoLLT solver(assembler.matrix());
        gsVector<> solVector = solver.solve(assembler.rhs());
        gsInfo << "Solved the system with PardisoLDLT solver in " << clock.stop() << "s.\n";
#else
        gsSparseSolver<>::SimplicialLDLT solver(assembler.matrix());
        gsVector<> solVector = solver.solve(assembler.rhs());
        gsInfo << "Solved the system with EigenLDLT solver in " << clock.stop() << "s.\n";
#endif
        // constructing displacement as an IGA function
        gsMultiPatch<> solution = assembler.patches();
        assembler.constructSolution(solVector, assembler.allFixedDofs(), solution);
        gsInfo << "Dofs: " << assembler.numDofs() << "\n\n";
        // constructing stress tensor
        gsPiecewiseFunction<> stresses;
        assembler.constructCauchyStresses(solution, stresses, stress_components::all_2D_vector);

        // constructing an IGA field (geometry + solution) for displacement
        gsField<> solutionField(assembler.patches(), solution);
        // constructing an IGA field (geometry + solution) for stresses
        gsField<> stressField(assembler.patches(), stresses, true);
        // analytical stresses
        gsField<> analyticalStressField(assembler.patches(), analyticalStresses, false);
        // creating a container to plot all fields to one Paraview file
        std::map<std::string, const gsField<>*> fields;
        fields["Deformation"] = &solutionField;
        fields["Stress"] = &stressField;
        fields["StressAnalytical"] = &analyticalStressField;
        fileName = baseName + util::to_string(refLoop);
        gsWriteParaviewMultiPhysics(fields, fileName, numPlotPoints, plotMesh);

        // --------------- adaptive refinement ---------------
        gsExprEvaluator<> ev;
        ev.setIntegrationElements(assembler.multiBasis());
        gsExprEvaluator<>::geometryMap Gm = ev.getMap(assembler.patches());
        gsExprEvaluator<>::variable is = ev.getVariable(stressField.fields());
        auto ms = ev.getVariable(analyticalStresses, Gm);

        // --------------- error computation ---------------
        real_t errorL2 = stressField.distanceL2(analyticalStresses, false);
        gsInfo << "L2 error of the solution is : " << errorL2 << "\n";
        real_t errorH1Semi = stressField.distanceH1(analyticalStresses, false);
        real_t errorH1 = math::sqrt(errorH1Semi * errorH1Semi + errorL2 * errorL2);
        gsInfo << "H1 error of the solution is : " << errorH1 << "\n";
        real_t errorH2Semi = stressField.distanceH1(analyticalStresses, false);
        real_t errorH2 = math::sqrt(errorH2Semi * errorH2Semi + errorH1Semi * errorH1Semi + errorL2 * errorL2);
        gsInfo << "H2 error of the solution is : " << errorH2 << "\n";
        ev.integralElWise((igrad(is, Gm) - igrad(ms)).sqNorm() * meas(Gm));
        gsInfo << "Energy Norm Error is : " << ev.integralElWise((igrad(is, Gm) - igrad(ms)).sqNorm() * meas(Gm)) << "\n\n";
        const std::vector<real_t>& eltErrs = ev.elementwise();
        std::vector<bool> elMarked(eltErrs.size());
        gsMarkElementsForRef(eltErrs, adaptRefCrit, adaptRefParam, elMarked);
        gsInfo << "Marked " << std::count(elMarked.begin(), elMarked.end(), true) << " elements.\n";
        gsRefineMarkedElements(basis, elMarked, 1);
        gsRefineMarkedElements(assembler.multiBasis(), elMarked, 1);
        basis.repairInterfaces(geometry.interfaces());
        assembler.multiBasis().repairInterfaces(assembler.patches().interfaces());

        // eval stress at the top of the circular cut
        gsMatrix<> A(2, 1);
        A << 1., 0.; // parametric coordinates for the isogeometric solution
        gsMatrix<> res;
        stresses.piece(1).eval_into(A, res);
        A << 0., 1.; // spatial coordinates for the analytical solution
        gsMatrix<> analytical;
        analyticalStresses.eval_into(A, analytical);
        gsInfo << "XX-stress at the top of the circle: " << res.at(0) << " (computed), " << analytical.at(0) << " (analytical)\n";
        gsInfo << "YY-stress at the top of the circle: " << res.at(1) << " (computed), " << analytical.at(1) << " (analytical)\n";
        gsInfo << "XY-stress at the top of the circle: " << res.at(2) << " (computed), " << analytical.at(2) << " (analytical)\n";
    }
    return 0;
}
