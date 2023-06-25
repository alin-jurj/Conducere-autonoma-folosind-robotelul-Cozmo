import { Image, View, Text, StyleSheet, Keyboard } from "react-native";
import Icon from "react-native-vector-icons/FontAwesome";
import { useNavigation } from '@react-navigation/native';
import { useState, useEffect, useContext } from "react";

function NavBar() {
    
    const navigation = useNavigation();

    const [isKeyboardVisible, setKeyboardVisible] = useState(false);

 useEffect(() => {
    const keyboardDidShowListener = Keyboard.addListener(
      'keyboardDidShow',
      () => {
        setKeyboardVisible(true); 
      }
    );
    const keyboardDidHideListener = Keyboard.addListener(
      'keyboardDidHide',
      () => {
        setKeyboardVisible(false); 
      }
    );

    return () => {
      keyboardDidHideListener.remove();
      keyboardDidShowListener.remove();
    };
  }, []);

    const goToHomeScreen = () => {
        navigation.navigate('ClientScreen');
        setCurrentScreen({homeScreen: true, settingsScreen: false, accountScreen: false});
    }

    const goToSettingsScreen = () => {
        navigation.navigate('BackgroundScreen');
        setCurrentScreen({homeScreen: false, settingsScreen: true, accountScreen: false});
    }

    const goToAccountScreen = () => {
        navigation.navigate('Account');
        setCurrentScreen({homeScreen: false, settingsScreen: false, accountScreen: true});
        
    }

    const [currentScreen, setCurrentScreen] = useState({
        homeScreen: true,
        settingsScreen: false,
        accountScreen: false
    })
    
    return(
        <View>
            {!isKeyboardVisible && 
                <View style={{height: 70, backgroundColor: "#3656A9", flexDirection: 'row', justifyContent: 'space-evenly', alignItems: 'center'}}>
                    <View style={{...styles.iconContainer, borderBottomColor: currentScreen.homeScreen == true ? '#fff' : '#3656A9'}}>
                        <Icon name="home" color={'#fff'} size={25} onPress={goToHomeScreen}></Icon>
                    </View>
                    <View style={{...styles.iconContainer, borderBottomColor: currentScreen.settingsScreen == true ? '#fff' : '#3656A9'}}>
                        <Icon name="gear" color={'#fff'} size={25} onPress={goToSettingsScreen}></Icon>
                    </View>
                    <View style={{...styles.iconContainer, borderBottomColor: currentScreen.accountScreen == true ? '#fff' : '#3656A9'}}>
                        <Icon name="user-circle-o" color={'#fff'} size={25} onPress={goToAccountScreen}></Icon>
                    </View>
                </View>
            }    
        </View>
    );
}

const styles = StyleSheet.create({
    iconContainer: {
        height: 35, 
        width: 35, 
        alignItems: 'center', 
        justifyContent: 'center', 
        borderColor: '#3656A9', 
        borderWidth: 3, 
        borderBottomLeftRadius: 4, 
        borderBottomRightRadius: 4
    }
  });

export default NavBar;