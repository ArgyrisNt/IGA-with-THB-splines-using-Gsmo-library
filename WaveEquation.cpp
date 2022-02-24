/*
Solution of Scalar Wave Propagation problem on an orthogonal domain using THB adaptive refinement.

This file is part of my Master Thesis.

Author : Argyrios Ntoumanis
*/

# include <gismo.h>
# include <gsAssembler/gsAdaptiveRefUtils.h>
# include "C:\Users\argir\gismo\src\gsAssembler\gsWaveEquation.h"

using namespace gismo;

int main(int argc, char* argv[])
{
    // --------------- Initialize Paraview ---------------
    bool plot = true;
    gsCmdLine cmd("Example for solving a wave propagation problem.");
    cmd.addSwitch("plot", "Plot the result in ParaView.", plot);
    try { cmd.getValues(argc, argv); }
    catch (int rv) { return rv; }

    // --------------- read geometry from file ---------------
    gsFunctionExpr<> f("0", 2);
    gsMultiPatch<> mp(*gsNurbsCreator<>::BSplineRectangle(0.0, 0.0, 6.0, 3.0));
    mp.computeTopology();
    gsTensorBSpline<2, real_t>* geo = dynamic_cast<gsTensorBSpline<2, real_t> *>(&mp.patch(0));
    gsTensorBSplineBasis<2, real_t> tbb = geo->basis();
    gsTHBSplineBasis<2, real_t> THB(tbb);
    gsTHBSpline<2, real_t> THB_patches(tbb, geo->coefs());
    gsMultiBasis<real_t> bases(THB);
    gsMultiPatch<real_t> patches(THB_patches);
    gsMultiPatch<real_t> newpatches = patches;
    patches.computeTopology();

    // --------------- add boundary conditions ---------------
    gsBoundaryConditions<> bcInfo;
    gsFunctionExpr<> gD("0", 2);
    gsFunctionExpr<> gD2("0.1", 2);
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, &gD);
    bcInfo.addCondition(0, boundary::east, condition_type::neumann, &gD2);
    bcInfo.addCondition(0, boundary::north, condition_type::neumann, &gD);
    bcInfo.addCondition(0, boundary::south, condition_type::neumann, &gD);

    // ------------ apply initial uniform refinement --------------  
    int numRefine = 4; // 0
    //bases.degreeElevate();
    //bases.degreeDecrease();
    for (int i = 0; i < numRefine; ++i)
    {
        bases.uniformRefine();
        newpatches.uniformRefine();
    }
    gsInfo << "Size of THB basis is: " << bases.basis(0).size() << "\n\n";
    gsInfo << "Degree of THB basis is " << bases.basis(0).degree(0) << "\n\n";
 
    //gsInfo << "Element's length is between " << bases.basis(0).getMinCellLength() << " and "
        //<< bases.basis(0).getMaxCellLength() << "\n\n";

    // --------------- define Pde ---------------
    gsPoissonPde<> pde(patches, bcInfo, f);
    gsPoissonAssembler<> stationary(pde, bases);
    stationary.options().setInt("DirichletValues", dirichlet::l2Projection);
    stationary.options().setInt("DirichletStrategy", dirichlet::elimination);
    stationary.options().setInt("InterfaceStrategy", iFace::glue);
    gsWaveEquation<real_t> assembler(stationary);
    real_t theta = 0.0; // 0 for first step, 1 for the rest
    assembler.setTheta(theta);

    // --------------- preliminaries (time step, solver, etc) ---------------
    real_t endTime = 48; // 24
    int numSteps = 4800; // 24 // 2400 
    //gsSparseSolver<>::BiCGSTABILUT solver;
    gsSparseSolver<>::BiCGSTABDiagonal solver;
    assembler.assemble();
    gsMatrix<> Sol, PrSol, Rhs;
    int ndof = assembler.numDofs(); 
    Sol.setZero(ndof, 1); // Initial solution u(0)
    PrSol = Sol;
    real_t Dt = endTime / numSteps;
    const std::string baseName("wave_eq_solution");
    gsParaviewCollection collection(baseName);
    gsParaviewCollection collection_mesh("mesh_" + baseName);
    std::string fileName;
    gsField<> sol = stationary.constructSolution(Sol);
    gsField<> prsol = stationary.constructSolution(PrSol);
    gsMultiBasis<real_t> bases2(THB);
    gsMatrix<> Sol2;
    gsSparseMatrix<> old_mass = assembler.mass();
    gsMatrix<real_t> u(2, 3);
    u(0, 0) = 1.0; u(1, 0) = 0.5; u(0, 1) = 0.5;
    u(1, 1) = 0.5; u(0, 2) = 0.25; u(1, 2) = 0.5;
    std::vector<real_t> exact1, exact2, exact3;

    for (int i = 1; i <= numSteps; ++i) // for all timesteps
    {
        if ((i == 1) || (i % (numSteps/16) == 0))
        {
            gsInfo << "\n------------------------------- Solving timestep " << i * Dt << " -------------------------------\n\n";
        }

        if (i == 2)
        {
            theta = 1.0;
            assembler.setTheta(theta);
        }

        //-------------------- COARSEN --------------------//

        if ((i == 1) || (i % (numSteps / 16) == 0))
        {
            bases2 = bases;
            newpatches = patches;
        }

        //-------------------- SOLVE --------------------//

        assembler.nextTimeStep(Sol, PrSol, Dt, old_mass);
        Sol2 = solver.compute(assembler.matrix()).solve(assembler.rhs());
        gsField<> sol2 = stationary.constructSolution(Sol2);

        // --------------- evaluate approx solution on specific points ---------------
        std::vector<gsMatrix<real_t>> solutions;
        solutions.push_back(sol2.value(u, 0));

        // --------------- evaluate exact solution on first point ---------------
        if ((0. < (i * Dt)) & ((i * Dt) <= 12.))
        {
            exact1.push_back(0.1 * i * Dt);
        }
        else if ((12. < (i * Dt)) & ((i * Dt) <= 24.))
        {
            exact1.push_back(0.1 * i * Dt - 0.2 * (i * Dt - 12));
        }

        // --------------- evaluate exact solution on second point ---------------
        if ((0. < (i * Dt)) & ((i * Dt) <= 3.))
        {
            exact2.push_back(0.);
        }
        else if ((3. < (i * Dt)) & ((i * Dt) <= 9.))
        {
            exact2.push_back(0.1 * (i * Dt - 0.3));
        }
        else if ((9. < (i * Dt)) & ((i * Dt) <= 15.))
        {
            exact2.push_back(0.6);
        }
        else if ((15. < (i * Dt)) & ((i * Dt) <= 21.))
        {
            exact2.push_back(0.6 - 0.1 * (i * Dt - 15));
        }
        else if ((21. < (i * Dt)) & ((i * Dt) <= 24.))
        {
            exact2.push_back(0.);
        }

        // --------------- evaluate exact solution on third point ---------------
        if ((0. < (i * Dt)) & ((i * Dt) <= 4.5))
        {
            exact3.push_back(0.);
        }
        else if ((4.5 < (i * Dt)) & ((i * Dt) <= 7.5))
        {
            exact3.push_back(0.1 * (i * Dt - 4.5));
        }
        else if ((7.5 < (i * Dt)) & ((i * Dt) <= 16.5))
        {
            exact3.push_back(0.3);
        }
        else if ((16.5 < (i * Dt)) & ((i * Dt) <= 19.5))
        {
            exact3.push_back(0.3 - 0.1 * (i * Dt - 16.5));
        }
        else if ((19.5 < (i * Dt)) & ((i * Dt) <= 24.))
        {
            exact3.push_back(0.);
        }

        //-------------------- REFINE --------------------//

        if ((i == 1) || (i % (numSteps / 16) == 0))
        {
            int numRefinementLoops = -1; //4 //3
            MarkingStrategy adaptRefCrit = PUCA;
            const real_t adaptRefParam = 0.5; //0.5 //0.7

            for (int refLoop = 0; refLoop <= numRefinementLoops; refLoop++)
            {
                //gsInfo << "====== Loop " << refLoop << " of "
                //    << numRefinementLoops << " ======" << "\n";

                gsExprEvaluator<> ev;
                ev.setIntegrationElements(bases2);
                gsExprEvaluator<>::geometryMap Gm = ev.getMap(newpatches);
                gsExprEvaluator<>::variable is = ev.getVariable(sol2.fields());
                ev.integralElWise((ilapl(is, Gm)).sqNorm() * meas(Gm));
                const std::vector<real_t>& eltErrs = ev.elementwise();
                std::vector<bool> elMarked(eltErrs.size());
                gsMarkElementsForRef(eltErrs, adaptRefCrit, adaptRefParam, elMarked);
                //gsInfo << "Marked " << std::count(elMarked.begin(), elMarked.end(), true) << " elements.\n";
                gsRefineMarkedElements(bases2, elMarked, 1);
                gsRefineMarkedElements(newpatches, elMarked, 1);
                //bases2.uniformRefine();
                //newpatches.uniformRefine();

                gsPoissonPde<> pde2(newpatches, bcInfo, f);
                gsPoissonAssembler<> stationary2(pde2, bases2);
                stationary2.options().setInt("DirichletValues", dirichlet::l2Projection);
                stationary2.options().setInt("DirichletStrategy", dirichlet::elimination);
                stationary2.options().setInt("InterfaceStrategy", iFace::glue);

                //-------------------- PROJECT PRSOL/SOL ON NEW MESH --------------------//

                Sol.resize(stationary2.numDofs());
                PrSol.resize(stationary2.numDofs());
                const gsDofMapper& dm = stationary2.system().colMapper(0);
                for (int k = 0; k < bases2.basis(0).size(); k++)
                {
                    const index_t l = dm.index(k, 0);
                    if (dm.is_free_index(l))
                    {
                        Sol.row(l) = gsQuasiInterpolate<real_t>::localIntpl(bases2[0], sol.function(0), k);
                        PrSol.row(l) = gsQuasiInterpolate<real_t>::localIntpl(bases2[0], prsol.function(0), k);
                    }
                }

                //-------------------- SOLVE AGAIN --------------------//

                Sol2.setZero(stationary2.numDofs(), 1);
                assembler.sameStep(Sol, PrSol, Dt, stationary2);
                Sol2 = solver.compute(assembler.matrix()).solve(assembler.rhs());
                sol2 = stationary2.constructSolution(Sol2);
                // --------------- evaluate approx solution on specific points ---------------
                gsInfo << "Size of THB basis is: " << bases2.basis(0).size() << "\n";
                solutions.push_back(sol2.value(u, 0));

                if (refLoop == numRefinementLoops)
                {
                    //-------------------- OUTPUT --------------------//                   
                    fileName = baseName + util::to_string(i);
                    gsWriteParaview<>(sol2, fileName, 1000, true);
                    collection.addTimestep(fileName, i, "0.vts");
                    collection_mesh.addTimestep(fileName, i, "0_mesh.vtp");
                }
            }
        }

        //-------------------- OUTPUT --------------------//

        if ((i == 1) || (i % (numSteps / 16) == 0))
        {
            gsInfo << std::endl;
            gsInfo << std::setw(25) << "u(6.0,1.5)" << std::setw(20) << "u(3.0,1.5)" << std::setw(20) << "u(1.5,1.5)" << "\n\n";
            gsInfo << std::setw(0) << "exact" << std::setprecision(8) << std::setw(20) << exact1[i-1] << std::setprecision(8) << std::setw(20) <<
                exact2[i - 1] << std::setprecision(8) << std::setw(20) << exact3[i - 1] << std::endl;
            for (size_t temp = 0; temp < solutions.size(); temp++)
            {
                gsInfo << std::setw(0) << "ref " << temp << std::setprecision(8) << std::setw(20) << solutions[temp][0] << std::setprecision(8) <<
                    std::setw(20) << solutions[temp][1] << std::setprecision(8) << std::setw(20) << solutions[temp][2] << std::endl;
            }
            gsInfo << std::endl;
        }

        //-------------------- PROJECT PRSOL/SOL ON OLD MESH --------------------//

        Sol.resize(stationary.numDofs());
        PrSol.resize(stationary.numDofs());
        const gsDofMapper& dm = stationary.system().colMapper(0);
        for (int k = 0; k < bases.basis(0).size(); k++)
        {
            const index_t l = dm.index(k, 0);
            if (dm.is_free_index(l))
            {
                Sol.row(l) = gsQuasiInterpolate<real_t>::localIntpl(bases[0], sol2.function(0), k);
                PrSol.row(l) = gsQuasiInterpolate<real_t>::localIntpl(bases[0], sol.function(0), k);
            }
        }
        sol = stationary.constructSolution(Sol);
        prsol = stationary.constructSolution(PrSol);
    }

    if (plot)
    {
        collection.save();
        collection_mesh.save();
        gsFileManager::open("mesh_" + baseName + ".pvd");
    }

    return  EXIT_SUCCESS;
}