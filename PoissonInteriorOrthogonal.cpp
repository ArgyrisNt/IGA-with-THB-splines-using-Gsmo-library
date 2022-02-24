/*
Solution of Poisson problem with interior singularity on an orthogonal domain using THB adaptive refinement.

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
   gsCmdLine cmd("Example on solving a Poisson problem.");
   cmd.addSwitch("plot", "Create a ParaView visualization file with the solution", plot);
   try { cmd.getValues(argc, argv); }
   catch (int rv) { return rv; }

   // --------------- specify exact solution and right-hand-side ---------------  
   gsFunctionExpr<> f("if ((x - 0.25) ^ 2 + (y - 0.6) ^ 2 < 0.2 ^ 2, 1, 0)",2);
   gsFunctionExpr<> g("0", 2);
   gsInfo << "Source function " << f << "\n";
   gsInfo << "Exact solution " << g << "\n\n";

   // --------------- read geometry from file ---------------
   gsMultiPatch<> mp(*gsNurbsCreator<>::BSplineRectangle(0.0, 0.0, 2.0, 1.0));
   gsTensorBSpline<2, real_t>* geo = dynamic_cast<gsTensorBSpline<2, real_t> *>(&mp.patch(0));
   gsTensorBSplineBasis<2, real_t> tbb = geo->basis();
   tbb.setDegree(2);
   gsTHBSplineBasis<2, real_t> THB(tbb);
   gsTHBSpline<2, real_t> THB_patches(tbb, geo->coefs());
   gsMultiBasis<real_t> bases(THB);
   gsMultiPatch<real_t> patches(THB_patches);

   // --------------- add boundary conditions ---------------   
   gsBoundaryConditions<> bcInfo;
   for (gsMultiPatch<>::const_biterator
       bit = patches.bBegin(); bit != patches.bEnd(); ++bit)
   {
       bcInfo.addCondition(*bit, condition_type::dirichlet, &g);
   }

   // ------------ apply initial uniform refinement --------------  
   //bases.degreeElevate();
   //bases.degreeElevate();
   int numInitUniformRefine = 2;//7
   for (int i = 0; i < numInitUniformRefine; ++i)
       bases.uniformRefine();
   gsInfo << "Size of THB basis is: " << bases.basis(0).size() << "\n\n";

   // --------------- set up adaptive refinement loop ---------------
   int numRefinementLoops = 7; //0
   MarkingStrategy adaptRefCrit = PUCA;
   const real_t adaptRefParam = 0.85;
   //MarkingStrategy adaptRefCrit = GARU;
   //const real_t adaptRefParam = 0.5;
   //MarkingStrategy adaptRefCrit = BULK;
   //const real_t adaptRefParam = 0.6;
   std::vector<real_t> exacts;

   //--------------- adaptive refinement loop ---------------
   for (int refLoop = 0; refLoop <= numRefinementLoops; refLoop++)
   {
       gsInfo << "====== Loop " << refLoop << " of "
       << numRefinementLoops << " ======" << "\n";

       // --------------- solving ---------------
       gsPoissonAssembler<real_t> PoissonAssembler(patches, bases, bcInfo, f);
       PoissonAssembler.options().setInt("DirichletValues", dirichlet::l2Projection);
       PoissonAssembler.assemble();
       gsSparseSolver<>::CGIdentity solver(PoissonAssembler.matrix());
       gsMatrix<> solVector = solver.solve(PoissonAssembler.rhs());
       gsMultiPatch<> sol;
       PoissonAssembler.constructSolution(solVector, sol);
       gsField<> solField(patches, sol);

       // --------------- error computation ---------------
       gsExprEvaluator<> ev;
       ev.setIntegrationElements(PoissonAssembler.multiBasis());
       gsExprEvaluator<>::geometryMap Gm = ev.getMap(patches);
       gsExprEvaluator<>::variable is = ev.getVariable(sol);
       auto ms = ev.getVariable(f, Gm);
       real_t error_exact = ev.integralElWise((ilapl(is, Gm) + ms).sqNorm() * meas(Gm));
       exacts.push_back(error_exact);

       //gsInfo << "Nonzeros entries of Stiffness matrix: " << PoissonAssembler.matrix().nonZeros() << "\n\n";

       const std::vector<real_t>& eltErrs = ev.elementwise();

       // --------------- adaptive refinement ---------------
       std::vector<bool> elMarked(eltErrs.size());
       gsMarkElementsForRef(eltErrs, adaptRefCrit, adaptRefParam, elMarked);
       //gsInfo << "Marked Elements: " << std::count(elMarked.begin(), elMarked.end(), true) << "\n";
       if (plot && refLoop == numRefinementLoops)
       {
           gsWriteParaview<>(solField, "PoissonInteriorOrthogonal_mesh", 1000, true);
           gsWriteParaview<>(solField, "PoissonInteriorOrthogonal_solution", 1000, false);
           gsInfo << "Exact error : ";
           for (size_t temp = 0; temp <= numRefinementLoops; temp++)
           {
               gsInfo << std::setprecision(8) << std::setw(15) << exacts[temp] << " | ";
           }
           gsInfo << "\n\n";
       }
       gsRefineMarkedElements(bases, elMarked, 1);
       bases.repairInterfaces(patches.interfaces());
       gsInfo << "Size of THB basis is: " << bases.basis(0).size() << "\n\n";

   }

   return EXIT_SUCCESS;

}
