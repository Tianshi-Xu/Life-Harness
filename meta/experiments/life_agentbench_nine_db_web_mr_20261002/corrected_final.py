"""Apply the same hash-bound runtime contract during final testing/reporting."""
import runtime_support
import final_reporting

if __name__ == "__main__":
    runtime_support.install()
    final_reporting.main()
