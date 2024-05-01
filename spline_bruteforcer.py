import sys

import numpy as np
import trajectory_planning_helpers as tph
from helper_funcs_glob.src import import_track
from optimize_globaltraj import ImportOptions

imp_opts = ImportOptions()
track_path = "./gs-short.csv"

reftrack_imp = import_track.import_track(
    imp_opts=imp_opts.__dict__,
    file_path=track_path,
    width_veh=0.32,
)

def compute_splines(k_reg: int, s_reg: int, stepsize_prep: float, stepsize_reg: float) -> int:
    reftrack_interp = tph.trajectory_planning_helpers.spline_approximation.spline_approximation(
        track=reftrack_imp,
        k_reg=k_reg,
        s_reg=s_reg,
        stepsize_prep=stepsize_prep,
        stepsize_reg=stepsize_reg,
        debug=True
    )

    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.trajectory_planning_helpers.calc_splines.calc_splines(
        path=refpath_interp_cl # type: ignore
    ) 

    normals_crossing = tph.trajectory_planning_helpers.check_normals_crossing.check_normals_crossing(
        track=reftrack_interp,
        normvec_normalized=normvec_normalized_interp,
        horizon=10
    )
    if normals_crossing:
        return 0

    return len(normvec_normalized_interp)

stepsize_prep_range = [0.05, 1.5]
stepsize_reg_range = [0.05, 1.5]

increment = 0.05

current_prep = stepsize_prep_range[0]
current_reg = stepsize_reg_range[0]

results = []

while current_prep <= stepsize_prep_range[1]:
    while current_reg <= stepsize_reg_range[1]:
        nb_splines = compute_splines(3, 1, current_prep, current_reg)
        print(f"{current_prep:.2f} - {current_reg:.2f}")

        if nb_splines > 0:
            results.append([current_prep, current_reg, nb_splines])
            break
            
        current_reg += increment

    current_prep += increment
    current_reg = stepsize_reg_range[0]

if len(results) == 0:
    print("No valid parameters combination found :(")
    sys.exit(1)

results.sort(key=lambda x: x[2], reverse=True)

print("==================== Results ====================")
print("[prep, reg, nb_splines]")
for result in results:
    print(f"{result[0]:.2f}, {result[1]:.2f}, {result[2]}")