import React, { useState } from 'react';
import styles from './index.module.css';
import Link from 'next/link';
import { ngolist } from '../../utils/ngo-list';
import { signUp } from '../../utils/firebase';
import { useRouter } from 'next/router';
import { NGOUserData } from '../../interfaces';

const Register = () => {
    const router = useRouter();
    const [email, setEmail] = useState('');
    const [pass, setPass] = useState('');
    const [userType, setUserType] = useState('');
    const [ngoSNo, setNgoSNo] = useState('');
    const [ngo, setNgo] = useState<NGOUserData>();
    const [userTypeError, setUserTypeError] = useState(false);
    const [restaurantData, setRestaurantData] = useState({
        name: '',
        address: '',
        latitude: '',
        longitude: '',
        pinCode: '',
        gstNumber: '',
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!userType) {
            setUserTypeError(true);
            return;
        }
        setUserTypeError(false);
        console.log('User type:', userType);

        try {
            // Register the user on Firebase
            let userData;
            if (userType === 'NGO') {
                userData = {
                    email: email,
                    userType,
                    userData: ngo,
                };
            } else {
                userData = {
                    email: email,
                    userType,
                    userData: restaurantData,
                };
            }

            const user = await signUp(email, pass, userData);
            console.log('User registered:', user);

            router.push('/');
        } catch (error) {
            console.error('Error registering user:', error);
        }
    };

    const handleRestaurantDataChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setRestaurantData({
            ...restaurantData,
            [e.target.name]: e.target.value,
        });
    };

    return (
        <div className={`${styles.lightMode}`}>
            <div className={`${styles.registerBox} `}>
                <div className={styles.formContainer}>
                    <h1 className={styles.title}>Register</h1>
                    <form onSubmit={handleSubmit} className={styles.form}>
                        <div>
                            <label htmlFor="email" className={styles.label}>
                                Email:
                            </label>
                            <input type="email" id="email" name="email" className={styles.input} onChange={(e) => setEmail(e.target.value)} required />
                        </div>
                        <div>
                            <label htmlFor="password" className={styles.label}>
                                Password:
                            </label>
                            <input type="password" id="password" name="password" className={styles.input} onChange={(e) => setPass(e.target.value)} required />
                        </div>
                        <div className={styles.tabContainer}>
                            <button
                                type="button"
                                className={`${styles.tab} ${userType === 'NGO' ? styles.selected : ''}`}
                                onClick={() => { setUserType('NGO'); setUserTypeError(false) }}
                            >
                                NGO
                            </button>
                            <button
                                type="button"
                                className={`${styles.tab} ${userType === 'Restaurant' ? styles.selected : ''}`}
                                onClick={() => { setUserType('Restaurant'); setUserTypeError(false) }}
                            >
                                Restaurant
                            </button>
                        </div>
                        {userTypeError && (
                            <p className={styles.error}>
                                Please select either NGO or Restaurant.
                            </p>
                        )}

                        {userType === 'NGO' && (
                            <div>
                                <label htmlFor="ngoSNo" className={styles.label}>
                                    NGO Name:
                                </label>
                                <select
                                    id="ngoSNo"
                                    name="ngoSNo"
                                    className={styles.select}
                                    value={ngoSNo}
                                    onChange={(e) => { setNgo(ngolist.find(o => o.SNo === parseInt(e.target.value)));setNgoSNo(e.target.value)}}
                                    required
                                >
                                    <option value="">Select an NGO</option>
                                    {ngolist.map((ngo) => (
                                        <option key={ngo.SNo} value={ngo.SNo}>
                                            {ngo["NGO Name"]}
                                        </option>
                                    ))}
                                </select>

                            </div>
                        )}
                        {userType === 'Restaurant' && (
                            <>
                                <div>
                                    <label htmlFor="name" className={styles.label}>
                                        Name:
                                    </label>
                                    <input
                                        type="text"
                                        id="name"
                                        name="name"
                                        className={styles.input}
                                        value={restaurantData.name}
                                        onChange={handleRestaurantDataChange}
                                        required
                                    />
                                </div>
                                <div>
                                    <label htmlFor="address" className={styles.label}>
                                        Address:
                                    </label>
                                    <input
                                        type="text"
                                        id="address"
                                        name="address"
                                        className={styles.input}
                                        value={restaurantData.address}
                                        onChange={handleRestaurantDataChange}
                                        required
                                    />
                                </div>
                                <div>
                                    <label htmlFor="latitude" className={styles.label}>
                                        Latitude:
                                    </label>
                                    <input
                                        type="text"
                                        id="latitude"
                                        name="latitude"
                                        className={styles.input}
                                        value={restaurantData.latitude}
                                        onChange={handleRestaurantDataChange}
                                        required
                                    />
                                </div>
                                <div>
                                    <label htmlFor="longitude" className={styles.label}>
                                        Longitude:
                                    </label>
                                    <input
                                        type="text"
                                        id="longitude"
                                        name="longitude"
                                        className={styles.input}
                                        value={restaurantData.longitude}
                                        onChange={handleRestaurantDataChange}
                                        required
                                    />
                                </div>
                                <div>
                                    <label htmlFor="pinCode" className={styles.label}>
                                        PinCode:
                                    </label>
                                    <input
                                        type="text"
                                        id="pinCode"
                                        name="pinCode"
                                        className={styles.input}
                                        value={restaurantData.pinCode}
                                        onChange={handleRestaurantDataChange}
                                        required
                                    />
                                </div>
                                <div>
                                    <label htmlFor="gstNumber" className={styles.label}>
                                        GST Number:
                                    </label>
                                    <input
                                        type="text"
                                        id="gstNumber"
                                        name="gstNumber"
                                        className={styles.input}
                                        value={restaurantData.gstNumber}
                                        onChange={handleRestaurantDataChange}
                                        required
                                        pattern="^(\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d{1}[A-Z]{1}\d{1})$"
                                        title="Enter a valid GST number (15 characters: 2 digits, 5 uppercase letters, 4 digits, 1 uppercase letter, 1 digit, 1 uppercase letter, 1 digit)"
                                    />
                                </div>
                            </>
                        )}
                        <button type="submit" className={styles.submitButton}>
                            Register
                        </button>
                        <div className={styles.loginLinkContainer}>
                            <span>Already have an account?</span>
                            <Link href="/login">
                                <span className={styles.loginLink}>Login</span>
                            </Link>
                        </div>
                        <div className={styles.homeLinkContainer}>
                            <Link href="/">
                                <span className={styles.homeLink}>Home</span>
                            </Link>
                        </div>
                    </form>
                </div>
            </div>
            <style jsx global>
                {`
                    :root {
                        /* Color palette */
                        --primary-color: #61876E;
                        --secondary-color: #21241f;
                        --tertiary-color: #EAE7B1;
                        --background-color: #3C6255;
                        --text-color: #FFFFFF;
                        --input-background-color: #A6BB8D;
                        --border-color: #EAE7B1;
                        --error-color: #E74C3C;
                    }
                    body {
                        background-color: var(--background-color);
                        color: var(--text-color);
                        font-family: Arial, sans-serif;
                    }
                `}
            </style>
        </div>
    );
};

export default Register;
