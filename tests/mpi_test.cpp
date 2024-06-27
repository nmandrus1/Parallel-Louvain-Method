#include <gtest/gtest.h>
#include <vector>
#include "graph.h"
#include <mpi.h>


class MPIEnvironment : public ::testing::Environment {
public:
  int *argc;
  char ***argv;

  MPIEnvironment(int *_argc, char ***_argv) : argc(_argc), argv(_argv) {}

  virtual void SetUp() {
    int mpiError = MPI_Init(argc, argv);
    ASSERT_FALSE(mpiError);
  }

  virtual void TearDown() {
    int mpiError = MPI_Finalize();
    ASSERT_FALSE(mpiError);
  }
};


int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment(&argc, &argv));
  return RUN_ALL_TESTS();
}
