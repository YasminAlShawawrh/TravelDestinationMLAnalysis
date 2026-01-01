"""
Main Runner Script for Tasks 3, 4, and 5
Executes all tasks sequentially
ENCS5341 - Assignment 3
"""
import sys
import os

def main():
    """Run all tasks in sequence"""
    print("=" * 70)
    print("TRAVEL DESTINATION ML ANALYSIS - TASKS 3, 4, 5")
    print("=" * 70)
    print("\nThis script will run:")
    print("  - Task 3: Baseline KNN Model")
    print("  - Task 4: Advanced ML Models")
    print("  - Task 5: Hyperparameter Tuning")
    print("\nNote: Make sure you have:")
    print("  - cleaned_data.csv (from Task 2)")
    print("  - All required packages installed (see requirements.txt)")
    print("\n" + "=" * 70)
    
    response = input("\nDo you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        return
    
    # Check if cleaned data exists
    if not os.path.exists('cleaned_data.csv'):
        print("\nERROR: cleaned_data.csv not found!")
        print("Please run Task 2 preprocessing first.")
        return
    
    # Run Task 3
    print("\n" + "=" * 70)
    print("RUNNING TASK 3: BASELINE KNN MODEL")
    print("=" * 70)
    try:
        from Task3_Baseline_KNN import main as task3_main
        task3_main()
    except Exception as e:
        print(f"\nERROR in Task 3: {str(e)}")
        print("Continuing to next task...")
    
    # Run Task 4
    print("\n" + "=" * 70)
    print("RUNNING TASK 4: ADVANCED ML MODELS")
    print("=" * 70)
    try:
        from Task4_Advanced_Models import main as task4_main
        task4_main()
    except Exception as e:
        print(f"\nERROR in Task 4: {str(e)}")
        print("Continuing to next task...")
    
    # Run Task 5
    print("\n" + "=" * 70)
    print("RUNNING TASK 5: HYPERPARAMETER TUNING")
    print("=" * 70)
    try:
        from Task5_Hyperparameter_Tuning import main as task5_main
        task5_main()
    except Exception as e:
        print(f"\nERROR in Task 5: {str(e)}")
        print("Task execution completed with errors.")
    
    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETED!")
    print("=" * 70)
    print("\nOutput directories created:")
    print("  - task3_outputs/")
    print("  - task4_outputs/")
    print("  - task5_outputs/")
    print("\nCheck these directories for results, visualizations, and reports.")


if __name__ == "__main__":
    main()

