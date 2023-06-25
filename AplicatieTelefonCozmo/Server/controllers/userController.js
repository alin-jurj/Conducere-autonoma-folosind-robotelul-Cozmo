const passanger = require('../models/Passanger')
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const express = require('express');
const app=express()
const getPassangers = async(req, res) => {
    const passangers = await passanger.find()

    res.status(200).json(passangers);
};
 
const loginUser = async(req, res) => {
  
    const { username, password} = req.body;

    var user = await passanger.findOne({username:username});

    if(!user)
         user = await driver.findOne({username:username});
    else
        if(!user)    
            return res.status(400).send('Username is wrong.');

    const validPassword = await bcrypt.compare(password, user.password);

    if(!validPassword)
        return res.status(400).send('password is wrong');
    
    if (user instanceof passanger)
        role  = 'Passanger'

    const token = jwt.sign({_id: user._id, role: role, username: username, email: user.email, pictureUrl: user.pictureUrl}, process.env.TOKEN_SECRET)

    try{
        res.header('auth-token',token).status(200).send(token);
    }catch(error){
        res.status(400).json({message: error});
    }
};

const signupPassenger = async(req,res) => {
    const { username, email, password, pictureUrl ,token} =  await req.body;

    const salt = await bcrypt.genSalt(10);
    const encryptedPassword = await bcrypt.hash(password,salt)

    try{
        const oldUser = await passanger.findOne({username:username});

        if(oldUser){
            res.send({error:"User Exists"});
        }
        else
            {       
                await passanger.create({
                    username,
                    email,
                    password: encryptedPassword,
                    pictureUrl,
                    token
                    });
                res.send({status: "ok"});
            }
    }catch(error){
        
        res.send({status:"error"});
        console.log(error);
    }
};

module.exports = {
    loginUser,
    getPassangers,
    signupPassenger,
};