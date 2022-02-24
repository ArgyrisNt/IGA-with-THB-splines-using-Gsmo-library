/*
Solution of Laplace problem with boundary singularity on an L shape domain using THB adaptive refinement.

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
  gsCmdLine cmd("Example on solving a Laplace problem.");
  cmd.addSwitch("plot", "Create a ParaView visualization file with the solution", plot);
  try { cmd.getValues(argc, argv); }
  catch (int rv) { return rv; }

  // --------------- specify exact solution and right-hand-side ---------------  
  gsFunctionExpr<> g("if( y>0, ( (x^2+y^2)^(1.0/3.0) )*sin( (2*atan2(y,x) - pi)/3.0 ),( (x^2+y^2)^(1.0/3.0) )*sin( (2*atan2(y,x)+3*pi)/3.0 )) ", 2);
  gsFunctionExpr<> f("0", 2);
  gsInfo << "Source function " << f << "\n";
  gsInfo << "Exact solution " << g << "\n\n";

  // --------------- read geometry from file ---------------
  gsMultiPatch<real_t> patches;
  std::string fileSrc("planar/lshape2d_3patches_thb.xml");
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
  int numInitUniformRefine = 0; //8
  for (int i = 0; i < numInitUniformRefine; ++i)
      bases.uniformRefine();
  gsInfo << "Size of THB basis is: " << bases.basis(0).size() << "\n\n";

  // --------------- set up adaptive refinement loop ---------------
  int numRefinementLoops = 6; //0
  MarkingStrategy adaptRefCrit = PUCA;
  const real_t adaptRefParam = 0.9; //0.3 //0.6
  //MarkingStrategy adaptRefCrit = GARU;
  //const real_t adaptRefParam = 0.1; //0.6 //0.9
  //MarkingStrategy adaptRefCrit = BULK;
  //const real_t adaptRefParam = 0.6; //0.3 //0.9
  std::vector<real_t> L2s;
  std::vector<real_t> H1s;
  std::vector<real_t> H2s;
  std::vector<real_t> energys;

  //--------------- adaptive refinement loop ---------------
  for (int refLoop = 0; refLoop <= numRefinementLoops; refLoop++)
  {
      gsInfo << "====== Loop " << refLoop << " of "
          << numRefinementLoops << " ======" << "\n";

      // --------------- solving ---------------
      gsPoissonAssembler<real_t> PoissonAssembler(patches, bases, bcInfo, f);
      PoissonAssembler.options().setInt("DirichletValues", dirichlet::l2Projection);
      PoissonAssembler.assemble();
      gsSparseSolver<>::CGDiagonal solver(PoissonAssembler.matrix());
      gsMatrix<> solVector = solver.solve(PoissonAssembler.rhs());
      gsMultiPatch<> sol;
      PoissonAssembler.constructSolution(solVector, sol);
      gsField<> solField(patches, sol);

      // --------------- error computation ---------------
      gsExprEvaluator<> ev;
      ev.setIntegrationElements(PoissonAssembler.multiBasis());
      gsExprEvaluator<>::geometryMap Gm = ev.getMap(patches);
      gsExprEvaluator<>::variable is = ev.getVariable(sol);
      auto ms = ev.getVariable(g, Gm);
      real_t error_energy = ev.integralElWise((igrad(is, Gm) - igrad(ms)).sqNorm() * meas(Gm));

      //gsInfo << "Nonzeros entries of Stiffness matrix: " << PoissonAssembler.matrix().nonZeros() << "\n\n";

      real_t errorL2 = solField.distanceL2(g, false);
      L2s.push_back(errorL2);
      real_t errorH1Semi = solField.distanceH1(g, false);
      real_t errorH1 = math::sqrt(errorH1Semi * errorH1Semi + errorL2 * errorL2);
      H1s.push_back(errorH1);
      real_t errorH2Semi = solField.distanceH1(g, false);
      real_t errorH2 = math::sqrt(errorH2Semi * errorH2Semi + errorH1Semi * errorH1Semi + errorL2 * errorL2);
      H2s.push_back(errorH2);
      energys.push_back(error_energy);
      const std::vector<real_t>& eltErrs = ev.elementwise();

      // --------------- adaptive refinement ---------------
      std::vector<bool> elMarked(eltErrs.size());
      gsMarkElementsForRef(eltErrs, adaptRefCrit, adaptRefParam, elMarked);
      //gsInfo << "Marked Elements: " << std::count(elMarked.begin(), elMarked.end(), true) << "\n";
      if (plot && refLoop == numRefinementLoops)
      {
          gsWriteParaview<>(solField, "LaplaceBoundaryLshape_mesh", 1000, true);
          gsWriteParaview<>(solField, "LaplaceBoundaryLshape_solution", 1000, false);
          gsInfo << "L2 : ";
          for (size_t temp = 0; temp <= numRefinementLoops; temp++)
          {
               gsInfo << std::setprecision(8) << std::setw(15) << L2s[temp] << " | ";
          }
          gsInfo << "\nH1 : ";
          for (size_t temp = 0; temp <= numRefinementLoops; temp++)
          {
              gsInfo << std::setprecision(8) << std::setw(15) << H1s[temp] << " | ";
          }
          gsInfo << "\nH2 : ";
          for (size_t temp = 0; temp <= numRefinementLoops; temp++)
          {
              gsInfo << std::setprecision(8) << std::setw(15) << H2s[temp] << " | ";
          }
          gsInfo << "\nEn : ";
          for (size_t temp = 0; temp <= numRefinementLoops; temp++)
          {
              gsInfo << std::setprecision(8) << std::setw(15) << energys[temp] << " | ";
          }
          gsInfo << "\n\n";
      }
      gsRefineMarkedElements(bases, elMarked, 1);
      bases.repairInterfaces(patches.interfaces());
      gsInfo << "Size of THB basis is: " << bases.basis(0).size() << "\n\n";
  }

  return EXIT_SUCCESS;

}
