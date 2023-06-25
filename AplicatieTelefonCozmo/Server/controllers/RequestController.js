const request = require('../models/Request')

const addRequest = async(req, res) => {
    const {passengerUsername, destination, passengerPicture, distance} = req.body;
    try
    {
         await request.create({
            passengerUsername,
            destination,
            passengerPicture,
            distance,
        })
        res.send({status:"ok"});
    }catch(error){
        
        res.send({status:"error"});
        console.log(error);
    }

}

const updateRequest = async(req, res) => {
    const id = req.query.id;
    const driverUsername = req.query.username;
    try
    {
        await request.updateOne({id: id}, {$set: {driverUsername: driverUsername}})
        res.send({status:"ok"});
    }catch(error){
        
        res.send({status:"error"});
        console.log(error);
    }
}

const getRequest = async(req, res) => {
    const resultedRequest = await request.find();

    res.status(200).json(resultedRequest);
};

const deleteRequest = async(req,res) =>{
    try {
        
        await request.deleteOne({passengerUsername:req.query.id})
        res.send('ok');
      
    } catch (error) {
      res.status(404).json({ message: error });
   
    }
    
};

module.exports = {
    addRequest,
    updateRequest,
    getRequest,
    deleteRequest,
}