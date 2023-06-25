import { StyleSheet, Text, View, ImageBackground } from 'react-native'
import Icon from "react-native-vector-icons/FontAwesome";
import React from 'react'
import {LoginContext} from '../App'
import { useContext } from 'react';
import { BackgroundContext } from '../App';


const AccountScreen = () => {
    const context = useContext(LoginContext);
    const backGroundContext = useContext(BackgroundContext);
    return (
    <View style={styles.container}>
        <ImageBackground source={{uri: backGroundContext.background}} resizeMode="stretch" style={styles.image}>
            <View style={{height: 100, width: 100, backgroundColor: 'white', borderRadius: 50}}>
               {/* {!context.loginDetails.photo} */}
                <Icon name="user-circle-o" color={'#3656A9'} size={100}></Icon>
            </View>

            <View style={styles.userInfo}>
                <View>
                    <Text style={styles.text}>Username:</Text>
                    <Text style={styles.text}>Email:</Text>
                </View>
            
                <View style={{marginLeft: 20}}>
                    <Text style={styles.text}>{context.loginDetails.username}</Text>
                    <Text style={styles.text}>{context.loginDetails.email}</Text>
                </View>
            </View>
        </ImageBackground>
    </View>
  )
}

export default AccountScreen

const styles = StyleSheet.create({
    container: {
        flex: 1,
        alignItems: 'center',
    },
    image: {
        paddingTop: 150,
        width: 480,
        height: 820,
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingBottom: 100
    },
    userInfo: {
        height: 200,
        width: 350,
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'white',
        borderWidth: 5,
        borderColor: '#3656A9',
        borderRadius: 20,
        marginBottom: 200,
    },
    text: {
        color: '#3656A9',
        fontWeight: 'bold',
        fontSize: 16
    }
})