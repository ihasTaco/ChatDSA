# ChatDSA
ChatDSA is a Discord chatbot built with Python and OpenAI's GPT-3 language model. It can help users with common questions and provide support in a chat thread.

## Features
- **Intuitive chat interface**: Users can interact with the bot through a chat thread.
- **Support for multiple users**: The bot can handle multiple chat threads simultaneously, allowing it to support multiple users at once.
- **Keyword-based file lookup**: The bot can find and return files from a specified directory based on keywords provided by the user.
- **Conversation history tracking**: The bot can keep track of conversation history to generate more relevant responses.
## Getting Started
### Prerequisites
To run ChatDSA, you will need the following software installed on your system:

- Python 3
- Py-Cord
- Discord Bot Token
- OpenAI API key

### Installation
1. Clone this repository to your local machine.
2. Install the required packages using the following command:
``pip install -r requirements.txt``

### Usage
To use ChatDSA, you will need to set up a Discord bot and obtain an API key from OpenAI. Follow the instructions below to get started:

1. Create a new Discord bot and obtain its token. You can follow the instructions [here](https://discordpy.readthedocs.io/en/stable/discord.html).
2. Set the DISCORD_TOKEN environment variable to the token of your Discord bot.
3. Obtain an API key from OpenAI and set the OPENAI_API_KEY environment variable to your API key.
4. Run the bot using the following command:
``python bot.py``

## Contributing
If you would like to contribute to ChatDSA, feel free to submit a pull request.

## License
This project is licensed under the MPL 2.0 License - see the [LICENSE](https://github.com/ihasTaco/ChatDSA/blob/main/License) file for details.
