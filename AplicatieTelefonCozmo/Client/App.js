// import { StatusBar } from 'expo-status-bar';
// import { StyleSheet, Text, View } from 'react-native';
import 'react-native-gesture-handler';
import { StatusBar } from 'expo-status-bar';
import { Settings } from 'react-native';
import { StyleSheet, Text, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import {Proba} from './screens/Proba';
import LoginScreen from './screens/LoginScreen';
import RegisterScreen from './screens/RegisterScreen';

import AccountScreen from './screens/AccountScreen';
import NotYetImplemented from './screens/RobotCamera';
import BackgroundScreen from './screens/BackgroundScreen';
import ClientScreen from './screens/ClientScreen';
import NavBar from './Components/NavBar'
import { useState } from 'react';
import React from 'react';
import cozmo from './assets/cozmo.jpg'
import RobotCamera from './screens/RobotCamera';
const Stack = createNativeStackNavigator();
export const BackgroundContext = React.createContext();
export const LoginContext = React.createContext();
export default function App() {


  const [background, setBackground] = useState('https://m.media-amazon.com/images/I/B1+BUJw5c0S.png');
  const [loginDetails, setloginDetails] = useState('');

  return (
   
       <NavigationContainer>
    
    <LoginContext.Provider value={{loginDetails,setloginDetails}}> 
    <BackgroundContext.Provider value={{background, setBackground}}> 
    <Stack.Navigator initialRouteName='LoginScreen'>
    <Stack.Screen options = {{headerShown: false}} name="LoginScreen" component={LoginScreen} />
    <Stack.Screen options = {{headerShown:false}} name="Register" component={RegisterScreen} />
    <Stack.Screen options = {{headerShown: false}} name="Proba" component={Proba} />
    <Stack.Screen options = {{headerShown: false}} name="Account" component={AccountScreen} />
    <Stack.Screen options = {{headerShown: false}} name="RobotCamera" component={RobotCamera}/>
    <Stack.Screen options = {{headerShown: false}} name="BackgroundScreen" component={BackgroundScreen}/>
    <Stack.Screen options = {{headerShown: false}} name="ClientScreen" component={ClientScreen}/>
    </Stack.Navigator>
    </BackgroundContext.Provider>
    </LoginContext.Provider>
    
    {loginDetails!='' && <NavBar />} 
  
  </NavigationContainer>
  );
}

