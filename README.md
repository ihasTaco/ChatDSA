# ChatDSA
ChatDSA (or Discord Server Assistant) is a Discord chatbot built with Python and OpenAI's GPT-3.5-turbo language model. It can help users with common questions and provide support in a chat thread.
Using ChatGPT's API system messages as context, you can send the AI information about your discord, it's purpose, and what fun and interesting things you can do in your discord.
While ChatDSA's main purpose is to help new and old discord members get around your discord and community, and give users relevant information, it also still functions like ChatGPT, so members can use it to answer trivial and not so trivial questions.

## Disclaimer
Be aware that OpenAI's gpt-3.5-turbo api cost $0.002 / 1K tokens while it is cheaper than other api's that they offer, it can still racket up charges based on how many users will use it.
Find your account usage here: https://platform.openai.com/account/usage

## Features
- **Intuitive chat interface**: Users can interact with the bot just by opening a thread and typing into it.
- **Support for multiple users**: The bot can handle multiple chat threads simultaneously, allowing it to support multiple users at once.
- **Keyword-based file lookup**: The bot can find and return files from a specified directory based on keywords provided by the user.
- **Conversation history tracking**: The bot can keep track of conversation history to generate more relevant responses.
## Getting Started
### Prerequisites
To run ChatDSA, you will need the following software installed on your system:

- Python 3
- Discord Bot Token
- OpenAI API key

### Installation
1. Clone this repository to your local machine.
2. Install the required packages using the following command:
``pip install -r requirements.txt``

### Usage
To use ChatDSA, you will need to set up a Discord bot and obtain an API key from OpenAI. Follow the instructions below to get started:

1. [Create a new Discord bot and obtain its token](https://discord.com/developers/applications). You can follow the instructions [here](https://discordpy.readthedocs.io/en/stable/discord.html).
  - The bot will need the following permissions
    - Read Messages/View Channels
    - Send Messages
    - Create Private Threads
    - Send Messages in Threads
    - Manage Threads
    - Embed Links (optional)
    - Read Message History
    - Use Slash Commands
2. Set the CHATDSA_TOKEN environment variable to the token of your Discord bot.
3. [Obtain an API key from OpenAI](https://platform.openai.com/account/api-keys) and set the OPENAI_API_KEY environment variable to your API key.
4. Set the channel to send the embed.
5. Make context files (see context/)
6. Run ``python generate_keywords.py`` to generate keywords, or create your own in 'keywords.json'
7. Run the bot using the following command:
``python bot.py``

## Coming Soon
these are in no particular order, but i will be working on functionality before anything.

- ability to fetch multiple files to get more info on subjects (Done. In testing)
  - im gonna run some test on using smaller context files, as i keep hitting the '4096 size limit' and the bot is coded to delete conversation history to make room for new responses, so you lose most of the context file contents anyways
- ability to give members optional roles (i.e. color roles, notification roles, etc)
  - I want this to be an optional setting, but the user can ask, "can you give me @red" and it will add it. but I also want to have a restricted roles setting so some roles cant be added (like moderator/admin roles).
- get user and server information (username, users roles, server name, and total members) (Done, except for the below part)
  - This will be useful for the give role feature, the ai will see a user is missing a role to do a certain action and can ask, "it seems you are missing the required role, would you like me to add it?"
- Integration with ServerQuery for live server information, also add a setting in config.ini to enable/disable
  - it would be cool to have users be able to ask 'whats the player count on the unturned arid server?' and the ai responds with, 'the unturned arid server is currently online and has 15 players, the peak time of the server today was at 5:40pm with 22 players' or something like that
- Add settings for embed customizations
  - right now the embed is hard coded with the title, description, button name (being completely honest, the embed and button were afterthoughts, i was originally just gonna have a /chat command) you can change them, its just, it is harder to do for the average user
- I want the threads initial message to be an embed to show what the users can ask (customizable)
- multi-guild support

## Contributing
If you would like to contribute to ChatDSA, feel free to submit a pull request.

## License
This project is licensed under the MPL 2.0 License - see the [LICENSE](https://github.com/ihasTaco/ChatDSA/blob/main/License) file for details.
