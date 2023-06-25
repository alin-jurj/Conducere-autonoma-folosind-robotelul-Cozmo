const searchUsername = require('../data_access/searchUsername');
const bcrypt = require('bcryptjs')

const addPassangerService = async(data) => {

    const {username, email, password} = data;

    const usernameExist = await searchUsername.searchUsername(username);

    if(usernameExist)
        return new Error('Username already exists');
    
    const emailExist = await searchEmail.searchEmail(email);

    if(emailExist)
        return new Error('Email already exists');

    const salt =await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password,salt);

    const passanger = new Passanger({
        username,
        email,
        password: hashedPassword,
        token
    });

    return passanger;
};

module.exports.addPassangerService = addPassangerService;