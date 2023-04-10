import Link from 'next/link';
import { useState } from 'react';
import styles from './index.module.css';
import { getUserDetails, logIn } from '../../utils/firebase';
import { useRouter } from 'next/router';
import Loader from '../../components/Loader';

export default function Login() {
    const router = useRouter();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [rememberMe, setRememberMe] = useState(false);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (event) => {
        event.preventDefault();
        setIsLoading(true);
        console.log("Login Click")
        const { user } = await logIn(email, password);
        console.log('User Logged In:', user);
        const userDetails = (await getUserDetails()).data();
        console.log('User Details:', userDetails);
        if (userDetails.userType === "NGO") {
            router.push('/ngo');
        } else if (userDetails.userType === "Restaurant") {
            router.push('/restaurant');
        // } else {
        //     setError('Invalid User Type');
        }
        setIsLoading(false);
    };

    return (
        <div className={`${styles.loginBox} ${styles.lightMode}`}>
            <h1 className={styles.title}>Login</h1>
            {error && <div className={styles.error}>{error}</div>}
            <form onSubmit={handleSubmit}>
                <label className={styles.label} htmlFor="email">
                    Email:
                </label>
                <input
                    className={styles.input}
                    type="email"
                    id="email"
                    name="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    required
                />

                <label className={styles.label} htmlFor="password">
                    Password:
                </label>
                <input
                    className={styles.input}
                    type="password"
                    id="password"
                    name="password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    required
                />

                <label className={styles.checkboxLabel}>
                    <input
                        className={styles.checkbox}
                        type="checkbox"
                        checked={rememberMe}
                        onChange={(event) => setRememberMe(event.target.checked)}
                    />
                    Remember me
                </label>

                <button className={styles.submitButton} type="submit">
                    Login
                </button>
            </form>
            <div className={styles.loginLinkContainer}>
                <span>Don't have an account?</span>
                <Link href="/register">
                    <span className={styles.loginLink}>Register</span>
                </Link>
            </div>
            <div className={styles.homeLinkContainer}>
                <Link href="/">
                    <span className={styles.homeLink}>Home</span>
                </Link>
            </div>
            <style jsx global>
                {`
                    :root {
                        /* Color palette */
                        --primary-color: #61876E;
                        --secondary-color: #A6BB8D;
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
            {isLoading && (
                <div className={styles.overlay}>
                    <Loader />
                </div>
            )}
        </div>
    );
}
