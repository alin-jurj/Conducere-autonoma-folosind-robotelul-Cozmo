const Passanger = require('../models/Passanger');

const searchUsername = async(username) => {
    let usernameExist = await Passanger.findOne({username: username});

    if(usernameExist)
        return new Error('Username already exists');
    return null;
};
module.exports.searchUsername = searchUsername;