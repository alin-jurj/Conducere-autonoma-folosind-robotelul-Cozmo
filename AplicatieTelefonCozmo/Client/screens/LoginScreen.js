import { KeyboardAvoidingView,ImageBackground, StyleSheet, Text, TextInput, Touchable, TouchableOpacity, View } from 'react-native'
import React, { useState } from 'react'
import { createNativeStackNavigator } from '@react-navigation/native-stack'
import RegisterScreen from './RegisterScreen';
import jwt_decode from "jwt-decode";
import {LoginContext} from '../App'
import { useContext } from 'react';
import { BackgroundContext } from '../App';


 const Stack = createNativeStackNavigator();

function LoginScreen ({navigation}) {

     const logincontext = useContext(LoginContext); 
     const backGroundContext = useContext(BackgroundContext);

     const [username, setusername]  = useState('')
     const [password, setPassword]  = useState('')
    
    let handleLogin = () =>{
      fetch("http://10.0.2.2:8000/user/signin",{
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body:JSON.stringify({
            username,
            password
        })
      })
      .then(res=>res.text())
      .then( data=>{
        console.log(data);
        const j = jwt_decode(data)
        const role = j.role
        
        logincontext.setloginDetails(j);
        if(role == "Passanger")
        {
           navigation.navigate('ClientScreen')
        }
        if(role == "Driver")
        {
          navigation.navigate('Proba')
        }
      }).catch(error =>console.error(error))
      
    }
    

  return (
   <KeyboardAvoidingView
     style={styles.container}
     behavior="padding"
     >
    <ImageBackground source={{uri: backGroundContext.background}} resizeMode="stretch" style={[styles.image,styles.container]}> 
    <View style = {styles.inputContainer}>
        <TextInput
          placeholder='Username'
          value = {username}
          onChangeText={text => setusername(text)}
        style = {styles.input}
        />
         <TextInput
          placeholder='Password'
          value = {password}
          onChangeText={text => setPassword(text)}
        style = {styles.input}
        secureTextEntry
        />
    </View>

    <View style = {styles.buttonContainer}>
        <TouchableOpacity
           onPress={handleLogin}
           style= {styles.button}
        >
          <Text style = {styles.buttonText}>Login</Text>  
        </TouchableOpacity>

        <TouchableOpacity
           onPress={() => navigation.navigate('Register')}
           style= {[styles.button, styles.buttonOutline]}
        >
          <Text style = {styles.buttonOutlineText}>Register</Text>  
        </TouchableOpacity>

    </View>
     </ImageBackground> 
    </KeyboardAvoidingView>
  )
}

export default LoginScreen

const styles = StyleSheet.create({
    container:{
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    inputContainer: {
        width: '80%'
    },
    input:{
        backgroundColor: 'white',
        paddingHorizontal:15,
        paddingVertical:10,
        borderRadius:10,
        marginTop:5,
    },
    buttonContainer:{
        width:'60%',
        justifyContent:'center',
        alignItems: 'center',
        marginTop: 40,
    },
    button: {
      backgroundColor: '#0782F9',
      width: '100%',
      padding: 15,
      borderRadius: 10,
      alignItems: 'center',
    },
    buttonOutline:{
      backgroundColor:'white',
      marginTop:5,
      borderColor: '#0782F9',
      borderWidth: 2,
    },
    buttonText:{
      color: 'white',
      fontWeight: '700',
      fontSize: 16,
    },
    buttonOutlineText:{
      color: '#0782F9',
      fontWeight: '700',
      fontSize: 16,
    },
    image: {
      flex: 1,
      justifyContent: "center",
      width:410,
      height:900,
    }
})