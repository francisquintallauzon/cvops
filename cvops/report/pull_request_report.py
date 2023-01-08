from pathlib import Path
import re

# TODO : Move to config
LOG_PATH = Path("./data/logs")
PULL_REQUEST_REPORT_OUTPUT_PATH = Path("./pull_request_report.md")


def get_version_number_from_folder_name(folder_name: str) -> int:

    """
    Extract the number from a folder_name.  For instance, if folder_name is "version_0" then the function will return zero.
    If multiple numbers are found in folder_name, then the last number of the string is returned. If no nuber is found in
    folder name, then an error is raised.

        Parameters:
                folder_name (str): folder name.

        Returns:
                folder_number (int): folder number
    """

    folder_number = re.findall(r"\d+", str(folder_name))

    if len(folder_number) == 0:
        raise ValueError(f'In input "{folder_name}", no number was found.')

    folder_number = folder_number[-1]
    return folder_number


def create_training_report(image_path: str | Path, report_file_path: str | Path):
    """
    Creates an algorithm training experiment markdown report that includes the input image.

        Parameters:
                image_path (str or Path): Path to image.
                report_file_path (str or Path): Report file path.
    """

    report = f"# Training report\n![Validation Loss Per Step]({str(image_path)})"

    with open(report_file_path, "w", encoding="utf-8") as file:
        file.write(report)


def create_training_report_from_latest_logs(log_path: str | Path, report_file_path: str | Path):
    """
    Create a algorithm training report using markdown from latest logs found in log_path.
    This function expects that each traning experiemnt logs be in a log_path/version_x folder.

        Parameters:
                log_path (str or pathlib.Path): Path where training logs are found
                report_file_path (str or pathlib.Path): Report file name.

    """

    # Extract log folders that contain a "validation_loss_per_step.png" file
    # TODO : Move image name to config
    log_paths = [pth for pth in log_path.glob("*") if (pth / "validation_loss_per_step.png").exists()]

    # Select the latest log
    if len(log_paths) == 0:
        raise FileNotFoundError(
            f'In "{str(log_path)}", did not fine any log folders that contains a "validation_loss_per_step.png" file'
        )

    latest_log_path = sorted(log_paths, key=get_version_number_from_folder_name)[-1]

    # Create cml report
    # TODO : Move image name to config
    print(f'Creating report with "{str(latest_log_path)}"')
    create_training_report(latest_log_path / "validation_loss_per_step.png", report_file_path)


if __name__ == "__main__":
    create_training_report_from_latest_logs(LOG_PATH, PULL_REQUEST_REPORT_OUTPUT_PATH)
