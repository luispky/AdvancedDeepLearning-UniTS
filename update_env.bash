#!/bin/bash

# Usage: ./conda_env_manager.sh <action> <env_name>
# <action>: "export" or "update"
# <env_name>: the name of the conda environment

# Check if sufficient arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <action> <env_name>"
    echo "<action>: export or update"
    echo "<env_name>: name of the conda environment"
    exit 1
fi

ACTION=$1
ENV_NAME=$2
ENV_FILE="environment.yml"

# Perform the specified action
case "$ACTION" in
    export)
        echo "Exporting conda environment '$ENV_NAME' to $ENV_FILE..."
        conda env export --name "$ENV_NAME" > "$ENV_FILE"
        if [ $? -eq 0 ]; then
            echo "Environment exported successfully to $ENV_FILE."
        else
            echo "Failed to export environment '$ENV_NAME'."
            exit 1
        fi
        ;;
    update)
        echo "Updating conda environment '$ENV_NAME' using $ENV_FILE..."
        if [ -f "$ENV_FILE" ]; then
            conda env update --name "$ENV_NAME" --file "$ENV_FILE"
            if [ $? -eq 0 ]; then
                echo "Environment '$ENV_NAME' updated successfully."
            else
                echo "Failed to update environment '$ENV_NAME'."
                exit 1
            fi
        else
            echo "Environment file '$ENV_FILE' not found. Please provide an environment.yml file."
            exit 1
        fi
        ;;
    *)
        echo "Invalid action: $ACTION. Please use 'export' or 'update'."
        exit 1
        ;;
esac
