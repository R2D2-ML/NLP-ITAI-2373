from src.utils.global_ import clear_terminal
from src.workflow import Workflow
from src.multilingual.detect import language_check
from dotenv import load_dotenv

load_dotenv()

def main():

    workflow = Workflow()
    clear_terminal()
    print("News Analysis Agent")

    while True:
        query = input("\nWhat can I do for you?: ")

        # Check if the user wants to quit the agent or make another query
        if query.lower() in {"quit", "exit"}:
            break

        print("\n\nAwesome, i'll get right on that for you!\n\n")

        query = language_check(query)
        print(query)

        # If there is a query run the agent
        if query:
            result = workflow.run(query)
            print(f"\nLets see if I can help you out...")
            print("=" * 60)

                
if __name__ == "__main__":
    main()