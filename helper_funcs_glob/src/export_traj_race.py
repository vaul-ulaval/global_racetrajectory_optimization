import hashlib
import uuid

import numpy as np


def export_traj_race(file_paths: dict, traj_race: np.ndarray) -> None:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function is used to export the generated trajectory into a file. The generated files get an unique UUID and a
    hash of the ggv diagram to be able to check it later.

    Inputs:
    file_paths:     paths for input and output files {ggv_file, traj_race_export, traj_ltpl_export, lts_export}
    traj_race:      race trajectory [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2]
    """

    # create random UUID
    rand_uuid = str(uuid.uuid4())

    # hash ggv file with SHA1
    if "ggv_file" in file_paths:
        with open(file_paths["ggv_file"], "br") as fh:
            ggv_content = fh.read()
    else:
        ggv_content = np.array([])
    ggv_hash = hashlib.sha1(ggv_content).hexdigest()

    # write UUID and GGV hash into file
    with open(file_paths["traj_race_export"], "w") as fh:
        fh.write("# " + rand_uuid + "\n")
        fh.write("# " + ggv_hash + "\n")

    # export race trajectory
    header = "s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2"
    fmt = "%.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f"
    with open(file_paths["traj_race_export"], "ab") as fh:
        np.savetxt(fh, traj_race, fmt=fmt, header=header)


def export_traj_race_f110(file_paths: dict, traj_race: np.ndarray) -> None:
    """
    Created by:
    Steven Gong

    Documentation:
    This function is used to export the generated trajectory into a csv file.
    We only need the x,y and velocity profile in the case of the f110 pure pursuit.

    Inputs:
    file_paths:     paths for input and output files {ggv_file, traj_race_export, traj_ltpl_export, lts_export}
    traj_race:      race trajectory [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2]
    """

    # export race trajectory
    header = "x_m, y_m, vx_mps, psi_rad, ax_mps2, kappa_radpm, s_m"
    fmt = "%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f"
    with open(file_paths["traj_race_export"], "wb") as fh:
        np.savetxt(
            fh, traj_race[:, [1, 2, 5, 3, 6, 4, 0]], fmt=fmt, comments="", header=header
        )


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
